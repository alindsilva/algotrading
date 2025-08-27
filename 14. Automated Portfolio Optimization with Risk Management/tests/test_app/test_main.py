"""
Integration tests for the main IBKR application.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime, timedelta
import json

from src.app.main import IBKRApp
from src.core.exceptions import ConnectionError
from src.orders.manager import OrderType, OrderAction, TimeInForce


class TestIBKRAppInitialization:
    """Test IBKR application initialization."""
    
    def test_init_default_config(self):
        """Test initialization with default configuration."""
        app = IBKRApp()
        
        assert app.config is not None
        assert app.client is None
        assert app.storage is None
        assert app.analytics is None
        assert not app.running
        assert len(app.active_subscriptions) == 0
        assert len(app.background_tasks) == 0
    
    def test_init_custom_config(self, temp_dir):
        """Test initialization with custom configuration."""
        # Create a temporary config file
        config_path = temp_dir / "test_config.yaml"
        config_content = """
        host: "192.168.1.100"
        port: 7496
        client_id: 10
        database:
          path: "custom_db.sqlite"
        """
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        app = IBKRApp(config_path=str(config_path))
        
        assert app.config.host == "192.168.1.100"
        assert app.config.port == 7496
        assert app.config.client_id == 10


class TestIBKRAppLifecycle:
    """Test application lifecycle management."""
    
    @pytest.mark.asyncio
    async def test_start_and_stop(self, test_config, mock_ibapi):
        """Test basic start and stop functionality."""
        app = IBKRApp()
        app.config = test_config
        
        # Mock the client start method
        with patch('src.app.main.IBClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.start.return_value = True
            mock_client.is_connected = True
            mock_client_class.return_value = mock_client
            
            # Mock storage and analytics
            with patch('src.app.main.AsyncDataStorage') as mock_storage_class:
                mock_storage = AsyncMock()
                mock_storage_class.return_value = mock_storage
                
                with patch('src.app.main.PortfolioAnalytics') as mock_analytics_class:
                    mock_analytics = Mock()
                    mock_analytics_class.return_value = mock_analytics
                    
                    # Start the application
                    success = await app.start()
                    
                    assert success
                    assert app.running
                    assert app.client is not None
                    assert app.storage is not None
                    assert app.analytics is not None
                    
                    # Stop the application
                    await app.stop()
                    
                    assert not app.running
                    mock_client.stop.assert_called_once()
                    mock_storage.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_start_failure(self, test_config, mock_ibapi):
        """Test application start failure handling."""
        app = IBKRApp()
        app.config = test_config
        
        # Mock client start to fail
        with patch('src.app.main.IBClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.start.return_value = False  # Simulate connection failure
            mock_client_class.return_value = mock_client
            
            with patch('src.app.main.AsyncDataStorage') as mock_storage_class:
                mock_storage = AsyncMock()
                mock_storage_class.return_value = mock_storage
                
                # Start should fail
                success = await app.start()
                
                assert not success
                assert not app.running
    
    @pytest.mark.asyncio
    async def test_double_start(self, test_config, mock_ibapi):
        """Test handling of double start."""
        app = IBKRApp()
        app.config = test_config
        
        with patch('src.app.main.IBClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.start.return_value = True
            mock_client_class.return_value = mock_client
            
            with patch('src.app.main.AsyncDataStorage') as mock_storage_class, \
                 patch('src.app.main.PortfolioAnalytics'):
                
                mock_storage = AsyncMock()
                mock_storage_class.return_value = mock_storage
                
                # Start twice
                success1 = await app.start()
                success2 = await app.start()
                
                assert success1
                assert success2  # Should handle gracefully
                
                await app.stop()


class TestPortfolioSummary:
    """Test portfolio summary functionality."""
    
    @pytest.mark.asyncio
    async def test_get_portfolio_summary(self, test_config, sample_positions, 
                                       sample_account_values, mock_ibapi):
        """Test getting portfolio summary."""
        app = IBKRApp()
        app.config = test_config
        
        # Mock client with data
        mock_client = AsyncMock()
        mock_client.is_connected = True
        mock_client.connection_info = {
            'connected': True,
            'uptime_seconds': 3600
        }
        mock_client.get_positions.return_value = sample_positions
        mock_client.get_account_values.return_value = sample_account_values
        
        # Mock storage
        mock_storage = AsyncMock()
        mock_storage.get_portfolio_returns.return_value = None  # No historical data
        
        # Mock analytics
        mock_analytics = AsyncMock()
        mock_analytics.analyze_positions.return_value = {
            'total_positions': 3,
            'long_exposure': 15000.0,
            'short_exposure': -2000.0,
            'net_exposure': 13000.0,
            'largest_positions': [
                {'symbol': 'AAPL', 'value': 15000.0, 'pct_of_portfolio': 0.15}
            ]
        }
        
        app.client = mock_client
        app.storage = mock_storage
        app.analytics = mock_analytics
        
        # Get portfolio summary
        summary = await app.get_portfolio_summary()
        
        # Verify summary structure
        assert 'timestamp' in summary
        assert 'connection_status' in summary
        assert 'positions' in summary
        assert 'portfolio_metrics' in summary
        assert 'account_summary' in summary
        
        # Verify data
        assert summary['positions']['total_positions'] == 3
        assert summary['connection_status']['connected']
        assert 'net_liquidation' in summary['account_summary']
    
    @pytest.mark.asyncio
    async def test_get_portfolio_summary_not_connected(self, test_config):
        """Test portfolio summary when not connected."""
        app = IBKRApp()
        app.config = test_config
        
        # Mock disconnected client
        mock_client = Mock()
        mock_client.is_connected = False
        app.client = mock_client
        
        # Should raise ConnectionError
        with pytest.raises(ConnectionError):
            await app.get_portfolio_summary()


class TestMarketDataStreaming:
    """Test market data streaming functionality."""
    
    @pytest.mark.asyncio
    async def test_start_market_data_stream(self, test_config, mock_ibapi):
        """Test starting market data streaming."""
        app = IBKRApp()
        app.config = test_config
        
        # Mock client
        mock_client = AsyncMock()
        mock_client.is_connected = True
        
        # Create an async generator for streaming data
        async def mock_streaming_data():
            for i in range(5):
                yield {
                    'request_id': 1,
                    'price': 150.0 + i,
                    'size': 100,
                    'tick_type': 1,
                    'timestamp': datetime.now()
                }
        
        mock_client.get_streaming_data.return_value = mock_streaming_data()
        
        # Mock storage
        mock_storage = AsyncMock()
        
        app.client = mock_client
        app.storage = mock_storage
        app.running = True
        
        # Start streaming
        success = await app.start_market_data_stream("AAPL")
        
        assert success
        assert "AAPL" in app.active_subscriptions
        
        # Wait a bit for some data to be processed
        await asyncio.sleep(0.1)
        
        # Stop streaming
        await app.stop_market_data_stream("AAPL")
        
        assert "AAPL" not in app.active_subscriptions
    
    @pytest.mark.asyncio
    async def test_start_market_data_stream_already_active(self, test_config, mock_ibapi):
        """Test starting stream for already active symbol."""
        app = IBKRApp()
        app.config = test_config
        
        mock_client = AsyncMock()
        mock_client.is_connected = True
        app.client = mock_client
        
        # Mock an existing subscription
        app.active_subscriptions["AAPL"] = AsyncMock()
        
        # Should return True but not create new subscription
        success = await app.start_market_data_stream("AAPL")
        
        assert success
        assert len(app.active_subscriptions) == 1
    
    @pytest.mark.asyncio
    async def test_start_market_data_stream_not_connected(self, test_config):
        """Test starting stream when not connected."""
        app = IBKRApp()
        app.config = test_config
        
        mock_client = Mock()
        mock_client.is_connected = False
        app.client = mock_client
        
        with pytest.raises(ConnectionError):
            await app.start_market_data_stream("AAPL")


class TestRiskReport:
    """Test risk report generation."""
    
    @pytest.mark.asyncio
    async def test_get_risk_report(self, test_config, sample_returns, sample_positions, 
                                 sample_account_values, mock_ibapi):
        """Test generating risk report."""
        app = IBKRApp()
        app.config = test_config
        
        # Mock components
        mock_client = AsyncMock()
        mock_client.get_positions.return_value = sample_positions
        mock_client.get_account_values.return_value = sample_account_values
        
        mock_storage = AsyncMock()
        mock_storage.get_portfolio_returns.return_value = sample_returns
        
        mock_analytics = AsyncMock()
        
        # Mock portfolio metrics
        from src.analytics.portfolio import PortfolioMetrics
        import pandas as pd
        
        mock_metrics = PortfolioMetrics(
            total_return=0.15,
            annualized_return=0.12,
            cumulative_returns=pd.Series([1.0, 1.1, 1.15]),
            volatility=0.20,
            max_drawdown=0.08,
            sharpe_ratio=0.75,
            sortino_ratio=0.85,
            calmar_ratio=1.5,
            omega_ratio=1.2,
            var_95=0.03,
            cvar_95=0.045
        )
        
        mock_analytics.calculate_portfolio_metrics.return_value = mock_metrics
        mock_analytics.calculate_rolling_metrics.return_value = {
            'rolling_volatility': pd.Series([0.18, 0.19, 0.20]),
            'rolling_sharpe_ratio': pd.Series([0.7, 0.75, 0.8])
        }
        mock_analytics.stress_test_portfolio.return_value = {
            'market_crash_10pct': {
                'sharpe_ratio_change': -0.3,
                'max_drawdown_change': 0.05,
                'volatility_change': 0.05
            }
        }
        mock_analytics.analyze_positions.return_value = {
            'total_positions': 3,
            'largest_positions': [
                {'symbol': 'AAPL', 'pct_of_portfolio': 0.15}
            ]
        }
        
        app.client = mock_client
        app.storage = mock_storage
        app.analytics = mock_analytics
        
        # Generate risk report
        report = await app.get_risk_report(days=30)
        
        # Verify report structure
        assert 'timestamp' in report
        assert 'analysis_period_days' in report
        assert 'portfolio_metrics' in report
        assert 'rolling_metrics' in report
        assert 'stress_test_results' in report
        assert 'position_analysis' in report
        assert 'risk_alerts' in report
        
        # Verify specific data
        assert report['analysis_period_days'] == 30
        assert report['portfolio_metrics']['sharpe_ratio'] == 0.75
        assert 'market_crash_10pct' in report['stress_test_results']
    
    @pytest.mark.asyncio
    async def test_get_risk_report_insufficient_data(self, test_config, mock_ibapi):
        """Test risk report with insufficient data."""
        app = IBKRApp()
        app.config = test_config
        
        mock_storage = AsyncMock()
        mock_storage.get_portfolio_returns.return_value = None  # No data
        
        mock_analytics = Mock()
        
        app.storage = mock_storage
        app.analytics = mock_analytics
        
        report = await app.get_risk_report(days=30)
        
        assert 'error' in report
        assert 'Insufficient historical data' in report['error']


class TestBackgroundTasks:
    """Test background task functionality."""
    
    @pytest.mark.asyncio
    async def test_portfolio_metrics_update(self, test_config, sample_positions, 
                                          sample_account_values, mock_ibapi):
        """Test portfolio metrics background update."""
        app = IBKRApp()
        app.config = test_config
        
        # Mock components
        mock_client = AsyncMock()
        mock_client.is_connected = True
        mock_client.get_positions.return_value = sample_positions
        mock_client.get_account_values.return_value = sample_account_values
        
        mock_storage = AsyncMock()
        
        app.client = mock_client
        app.storage = mock_storage
        app.running = True
        
        # Manually call the update method
        await app._update_portfolio_metrics()
        
        # Verify that methods were called
        mock_client.get_positions.assert_called_once()
        mock_client.get_account_values.assert_called_once()
        
        # Should have updated last update time
        assert app.last_portfolio_update is not None


class TestApplicationStatus:
    """Test application status reporting."""
    
    def test_status_property(self, test_config):
        """Test status property."""
        app = IBKRApp()
        app.config = test_config
        app.running = True
        
        # Mock client
        mock_client = Mock()
        mock_client.is_connected = True
        app.client = mock_client
        
        # Add some mock subscriptions
        app.active_subscriptions = {"AAPL": Mock(), "GOOGL": Mock()}
        app.background_tasks = [Mock(), Mock(), Mock()]
        
        status = app.status
        
        assert status['running'] is True
        assert status['connected'] is True
        assert len(status['active_subscriptions']) == 2
        assert status['background_tasks'] == 3
        assert 'AAPL' in status['active_subscriptions']
        assert 'GOOGL' in status['active_subscriptions']


class TestRiskAlerts:
    """Test risk alert generation."""
    
    def test_generate_risk_alerts_high_volatility(self, test_config):
        """Test risk alerts for high volatility."""
        app = IBKRApp()
        app.config = test_config
        
        # Mock high volatility metrics
        from src.analytics.portfolio import PortfolioMetrics
        import pandas as pd
        
        metrics = PortfolioMetrics(
            total_return=0.15,
            annualized_return=0.12,
            cumulative_returns=pd.Series([1.0, 1.15]),
            volatility=0.35,  # High volatility
            max_drawdown=0.08,
            sharpe_ratio=0.75,
            sortino_ratio=0.85,
            calmar_ratio=1.5,
            omega_ratio=1.2,
            var_95=0.03,
            cvar_95=0.045
        )
        
        position_analysis = {
            'largest_positions': [
                {'symbol': 'AAPL', 'pct_of_portfolio': 0.15}
            ]
        }
        
        alerts = app._generate_risk_alerts(metrics, position_analysis)
        
        # Should have high volatility alert
        volatility_alerts = [alert for alert in alerts if alert['type'] == 'HIGH_VOLATILITY']
        assert len(volatility_alerts) == 1
        assert volatility_alerts[0]['severity'] == 'WARNING'
    
    def test_generate_risk_alerts_concentration_risk(self, test_config):
        """Test risk alerts for concentration risk."""
        app = IBKRApp()
        app.config = test_config
        
        from src.analytics.portfolio import PortfolioMetrics
        import pandas as pd
        
        metrics = PortfolioMetrics(
            total_return=0.15,
            annualized_return=0.12,
            cumulative_returns=pd.Series([1.0, 1.15]),
            volatility=0.15,  # Normal volatility
            max_drawdown=0.08,
            sharpe_ratio=0.75,
            sortino_ratio=0.85,
            calmar_ratio=1.5,
            omega_ratio=1.2,
            var_95=0.03,
            cvar_95=0.045
        )
        
        position_analysis = {
            'largest_positions': [
                {'symbol': 'AAPL', 'pct_of_portfolio': 0.35},  # High concentration
                {'symbol': 'GOOGL', 'pct_of_portfolio': 0.15}
            ]
        }
        
        alerts = app._generate_risk_alerts(metrics, position_analysis)
        
        # Should have concentration risk alert
        concentration_alerts = [alert for alert in alerts if alert['type'] == 'CONCENTRATION_RISK']
        assert len(concentration_alerts) == 1
        assert 'AAPL' in concentration_alerts[0]['message']


class TestTradingMethods:
    """Test trading order methods in IBKRApp."""
    
    @pytest.mark.asyncio
    async def test_place_buy_order(self, test_config, mock_ibapi):
        """Test placing a buy order."""
        app = IBKRApp()
        app.config = test_config
        
        # Mock order manager
        mock_order_manager = AsyncMock()
        mock_order_manager.place_order.return_value = {
            "status": "placed",
            "order_id": 123,
            "symbol": "AAPL",
            "action": "BUY",
            "quantity": 100
        }
        
        # Mock client
        mock_client = Mock()
        mock_client.is_connected = True
        
        app.order_manager = mock_order_manager
        app.client = mock_client
        
        result = await app.place_buy_order("AAPL", 100)
        
        assert result["status"] == "placed"
        assert result["order_id"] == 123
        assert result["symbol"] == "AAPL"
        mock_order_manager.place_order.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_place_sell_order(self, test_config, mock_ibapi):
        """Test placing a sell order."""
        app = IBKRApp()
        app.config = test_config
        
        mock_order_manager = AsyncMock()
        mock_order_manager.place_order.return_value = {
            "status": "placed",
            "order_id": 124,
            "symbol": "AAPL",
            "action": "SELL",
            "quantity": 50
        }
        
        mock_client = Mock()
        mock_client.is_connected = True
        
        app.order_manager = mock_order_manager
        app.client = mock_client
        
        result = await app.place_sell_order("AAPL", 50)
        
        assert result["status"] == "placed"
        assert result["order_id"] == 124
        assert result["action"] == "SELL"
    
    @pytest.mark.asyncio
    async def test_order_target_percent(self, test_config, mock_ibapi):
        """Test ordering to target percentage allocation."""
        app = IBKRApp()
        app.config = test_config
        
        mock_order_manager = AsyncMock()
        mock_order_manager.order_target_percent.return_value = {
            "status": "placed",
            "symbol": "AAPL",
            "target_percent": 0.25,
            "current_percent": 0.15,
            "action_taken": "BUY",
            "quantity": 50
        }
        
        mock_client = Mock()
        mock_client.is_connected = True
        
        app.order_manager = mock_order_manager
        app.client = mock_client
        
        result = await app.order_target_percent("AAPL", 0.25)
        
        assert result["status"] == "placed"
        assert result["target_percent"] == 0.25
        mock_order_manager.order_target_percent.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_order_target_quantity(self, test_config, mock_ibapi):
        """Test ordering to target quantity."""
        app = IBKRApp()
        app.config = test_config
        
        mock_order_manager = AsyncMock()
        mock_order_manager.order_target_quantity.return_value = {
            "status": "placed",
            "symbol": "AAPL",
            "target_quantity": 100,
            "current_quantity": 50,
            "quantity_to_trade": 50,
            "action": "BUY"
        }
        
        mock_client = Mock()
        mock_client.is_connected = True
        
        app.order_manager = mock_order_manager
        app.client = mock_client
        
        result = await app.order_target_quantity("AAPL", 100)
        
        assert result["status"] == "placed"
        assert result["target_quantity"] == 100
        mock_order_manager.order_target_quantity.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_order_target_value(self, test_config, mock_ibapi):
        """Test ordering to target value."""
        app = IBKRApp()
        app.config = test_config
        
        mock_order_manager = AsyncMock()
        mock_order_manager.order_target_value.return_value = {
            "status": "placed",
            "symbol": "AAPL",
            "target_value": 5000.0,
            "current_value": 3000.0,
            "value_to_trade": 2000.0,
            "quantity": 13,
            "action": "BUY"
        }
        
        mock_client = Mock()
        mock_client.is_connected = True
        
        app.order_manager = mock_order_manager
        app.client = mock_client
        
        result = await app.order_target_value("AAPL", 5000.0)
        
        assert result["status"] == "placed"
        assert result["target_value"] == 5000.0
        mock_order_manager.order_target_value.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_order_value(self, test_config, mock_ibapi):
        """Test ordering a fixed value."""
        app = IBKRApp()
        app.config = test_config
        
        mock_order_manager = AsyncMock()
        mock_order_manager.order_value.return_value = {
            "status": "placed",
            "symbol": "AAPL",
            "value": 1000.0,
            "quantity": 6,
            "action": "BUY",
            "estimated_price": 166.67
        }
        
        mock_client = Mock()
        mock_client.is_connected = True
        
        app.order_manager = mock_order_manager
        app.client = mock_client
        
        result = await app.order_value("AAPL", 1000.0)
        
        assert result["status"] == "placed"
        assert result["value"] == 1000.0
        mock_order_manager.order_value.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cancel_all_orders(self, test_config, mock_ibapi):
        """Test cancelling all orders."""
        app = IBKRApp()
        app.config = test_config
        
        mock_order_manager = AsyncMock()
        mock_order_manager.cancel_all_orders.return_value = {
            "status": "completed",
            "total_cancelled": 3,
            "cancelled_orders": [101, 102, 103]
        }
        
        mock_client = Mock()
        mock_client.is_connected = True
        
        app.order_manager = mock_order_manager
        app.client = mock_client
        
        result = await app.cancel_all_orders()
        
        assert result["status"] == "completed"
        assert result["total_cancelled"] == 3
        mock_order_manager.cancel_all_orders.assert_called_once()
    
    def test_cancel_order_by_id(self, test_config, mock_ibapi):
        """Test cancelling order by ID."""
        app = IBKRApp()
        app.config = test_config
        
        mock_client = Mock()
        mock_client.is_connected = True
        mock_client.cancel_order_by_id = Mock()
        
        app.client = mock_client
        
        result = app.cancel_order_by_id(123)
        
        assert result["status"] == "requested"
        assert result["order_id"] == 123
        mock_client.cancel_order_by_id.assert_called_once_with(123)
    
    @pytest.mark.asyncio
    async def test_update_order(self, test_config, mock_ibapi):
        """Test updating an order."""
        app = IBKRApp()
        app.config = test_config
        
        mock_order_manager = AsyncMock()
        mock_order_manager.update_order.return_value = {
            "status": "updated",
            "order_id": 123,
            "old_quantity": 100,
            "new_quantity": 150
        }
        
        mock_client = Mock()
        mock_client.is_connected = True
        
        app.order_manager = mock_order_manager
        app.client = mock_client
        
        result = await app.update_order(123, "AAPL", 150)
        
        assert result["status"] == "updated"
        assert result["order_id"] == 123
        mock_order_manager.update_order.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_pnl(self, test_config, mock_ibapi):
        """Test getting P&L data."""
        app = IBKRApp()
        app.config = test_config
        
        mock_client = AsyncMock()
        mock_client.is_connected = True
        mock_client.get_pnl.return_value = {
            "daily_pnl": 1250.75,
            "unrealized_pnl": 850.25,
            "realized_pnl": 400.50,
            "account": "DU123456"
        }
        
        app.client = mock_client
        
        result = await app.get_pnl("DU123456")
        
        assert result["daily_pnl"] == 1250.75
        assert result["account"] == "DU123456"
        mock_client.get_pnl.assert_called_once_with(account="DU123456")
    
    @pytest.mark.asyncio
    async def test_close_position(self, test_config, sample_positions, mock_ibapi):
        """Test closing a position."""
        app = IBKRApp()
        app.config = test_config
        
        # Mock position data - positive position (long)
        from src.core.types import PositionData
        positions = [
            PositionData(
                account="DU123456",
                symbol="AAPL",
                sec_type="STK",
                exchange="NASDAQ",
                currency="USD",
                position=100.0,  # Long 100 shares
                avg_cost=145.0,
                timestamp=datetime.now()
            )
        ]
        
        mock_client = AsyncMock()
        mock_client.is_connected = True
        mock_client.get_positions.return_value = positions
        
        mock_order_manager = AsyncMock()
        
        app.client = mock_client
        app.order_manager = mock_order_manager
        
        # Mock the place_sell_order method
        with patch.object(app, 'place_sell_order') as mock_place_sell:
            mock_place_sell.return_value = {
                "status": "placed",
                "symbol": "AAPL",
                "action": "SELL",
                "quantity": 100
            }
            
            result = await app.close_position("AAPL")
            
            assert result["status"] == "placed"
            assert result["close_position"] is True
            assert result["original_position"] == 100
            mock_place_sell.assert_called_once_with(
                symbol="AAPL",
                quantity=100,
                order_type="MKT",
                dry_run=False
            )
    
    @pytest.mark.asyncio
    async def test_close_position_no_position(self, test_config, mock_ibapi):
        """Test closing position when no position exists."""
        app = IBKRApp()
        app.config = test_config
        
        mock_client = AsyncMock()
        mock_client.is_connected = True
        mock_client.get_positions.return_value = []  # No positions
        
        app.client = mock_client
        
        result = await app.close_position("AAPL")
        
        assert result["status"] == "no_position"
        assert "No position found" in result["message"]


class TestPortfolioMethods:
    """Test portfolio management methods."""
    
    @pytest.mark.asyncio
    async def test_rebalance_portfolio(self, test_config, mock_ibapi):
        """Test portfolio rebalancing."""
        app = IBKRApp()
        app.config = test_config
        
        target_allocations = {
            "AAPL": 0.30,
            "GOOGL": 0.25,
            "MSFT": 0.20,
            "TSLA": 0.15
        }
        
        mock_order_manager = AsyncMock()
        mock_order_manager.rebalance_portfolio.return_value = {
            "summary": {
                "status": "completed",
                "total_symbols": 4,
                "orders_placed": 3,
                "dry_run": False
            },
            "results": {
                "AAPL": {"status": "placed", "action": "BUY", "quantity": 25},
                "GOOGL": {"status": "placed", "action": "SELL", "quantity": 10},
                "MSFT": {"status": "no_change"},
                "TSLA": {"status": "placed", "action": "BUY", "quantity": 5}
            }
        }
        
        mock_client = Mock()
        mock_client.is_connected = True
        
        app.order_manager = mock_order_manager
        app.client = mock_client
        
        result = await app.rebalance_portfolio(target_allocations)
        
        assert result["summary"]["status"] == "completed"
        assert result["summary"]["orders_placed"] == 3
        assert "AAPL" in result["results"]
        mock_order_manager.rebalance_portfolio.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_portfolio_allocations(self, test_config, mock_ibapi):
        """Test getting portfolio allocations."""
        app = IBKRApp()
        app.config = test_config
        
        from src.core.types import PositionData, AccountValue
        
        # Mock positions - Use proper dataclass format
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
                position=50.0,
                avg_cost=95.0,
                timestamp=datetime.now()
            )
        ]
        
        # Mock account values
        account_values = [
            AccountValue(
                key="NetLiquidation", 
                value="25000.0", 
                currency="USD",
                account="DU123456",
                timestamp=datetime.now()
            )
        ]
        
        mock_client = AsyncMock()
        mock_client.is_connected = True
        mock_client.get_positions.return_value = positions
        mock_client.get_account_values.return_value = account_values
        
        app.client = mock_client
        
        # Mock get_market_quote for current prices
        with patch.object(app, 'get_market_quote') as mock_quote:
            mock_quote.side_effect = [
                {"last": 150.0},  # AAPL
                {"last": 100.0}   # GOOGL
            ]
            
            allocations = await app.get_portfolio_allocations()
            
            assert "AAPL" in allocations
            assert "GOOGL" in allocations
            assert abs(allocations["AAPL"] - 0.60) < 0.01  # 15000/25000 = 0.6
            assert abs(allocations["GOOGL"] - 0.20) < 0.01  # 5000/25000 = 0.2
    
    @pytest.mark.asyncio
    async def test_suggest_rebalance(self, test_config, mock_ibapi):
        """Test rebalancing suggestions."""
        app = IBKRApp()
        app.config = test_config
        
        target_allocations = {
            "AAPL": 0.25,
            "GOOGL": 0.25,
            "MSFT": 0.25,
            "TSLA": 0.25
        }
        
        # Mock current allocations - AAPL overweight, TSLA underweight
        current_allocations = {
            "AAPL": 0.35,  # 10% overweight
            "GOOGL": 0.25,  # On target
            "MSFT": 0.25,   # On target
            "TSLA": 0.15    # 10% underweight
        }
        
        with patch.object(app, 'get_portfolio_allocations') as mock_get_allocations:
            mock_get_allocations.return_value = current_allocations
            
            suggestions = await app.suggest_rebalance(target_allocations, threshold=0.05)
            
            assert suggestions["needs_rebalance"] is True
            assert "AAPL" in suggestions["suggestions"]
            assert "TSLA" in suggestions["suggestions"]
            assert suggestions["suggestions"]["AAPL"]["action"] == "DECREASE"
            assert suggestions["suggestions"]["TSLA"]["action"] == "INCREASE"
            # GOOGL and MSFT should not be in suggestions (within threshold)
            assert "GOOGL" not in suggestions["suggestions"]
            assert "MSFT" not in suggestions["suggestions"]


class TestMarketDataMethods:
    """Test market data related methods."""
    
    @pytest.mark.asyncio
    async def test_get_market_quote(self, test_config, mock_ibapi):
        """Test getting market quote."""
        app = IBKRApp()
        app.config = test_config
        
        mock_client = AsyncMock()
        mock_client.is_connected = True
        mock_client.get_market_data.return_value = {
            "bid": 149.50,
            "ask": 150.50,
            "last": 150.00,
            "volume": 1000000,
            "high": 152.00,
            "low": 148.00,
            "close": 149.75
        }
        
        app.client = mock_client
        
        quote = await app.get_market_quote("AAPL")
        
        assert quote["symbol"] == "AAPL"
        assert quote["bid"] == 149.50
        assert quote["ask"] == 150.50
        assert quote["last"] == 150.00
        assert "timestamp" in quote
        mock_client.get_market_data.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_historical_data(self, test_config, mock_ibapi):
        """Test getting historical data."""
        app = IBKRApp()
        app.config = test_config
        
        historical_bars = [
            {
                "timestamp": "2024-01-01 09:30:00",
                "open": 148.00,
                "high": 152.00,
                "low": 147.50,
                "close": 150.00,
                "volume": 1000000
            },
            {
                "timestamp": "2024-01-02 09:30:00",
                "open": 150.00,
                "high": 153.00,
                "low": 149.00,
                "close": 152.00,
                "volume": 1200000
            }
        ]
        
        mock_client = AsyncMock()
        mock_client.is_connected = True
        mock_client.get_historical_data.return_value = historical_bars
        
        app.client = mock_client
        
        data = await app.get_historical_data("AAPL", "1 D", "1 day")
        
        assert len(data) == 2
        assert data[0]["open"] == 148.00
        assert data[1]["close"] == 152.00
        mock_client.get_historical_data.assert_called_once()


class TestErrorHandling:
    """Test error handling in trading methods."""
    
    @pytest.mark.asyncio
    async def test_trading_method_not_connected(self, test_config):
        """Test trading methods when not connected."""
        app = IBKRApp()
        app.config = test_config
        
        mock_client = Mock()
        mock_client.is_connected = False
        app.client = mock_client
        
        # Add a mock order manager so we get past the order manager check
        mock_order_manager = AsyncMock()
        app.order_manager = mock_order_manager
        
        # Test various methods raise ConnectionError when not connected
        with pytest.raises(ConnectionError):
            await app.place_buy_order("AAPL", 100)
        
        with pytest.raises(ConnectionError):
            await app.order_target_percent("AAPL", 0.25)
        
        with pytest.raises(ConnectionError):
            await app.cancel_all_orders()
        
        with pytest.raises(ConnectionError):
            await app.get_pnl()
    
    @pytest.mark.asyncio
    async def test_trading_method_no_order_manager(self, test_config):
        """Test trading methods when order manager is not initialized."""
        app = IBKRApp()
        app.config = test_config
        
        mock_client = Mock()
        mock_client.is_connected = True
        app.client = mock_client
        app.order_manager = None  # Not initialized
        
        with pytest.raises(RuntimeError, match="Order manager not initialized"):
            await app.place_buy_order("AAPL", 100)
        
        with pytest.raises(RuntimeError, match="Order manager not initialized"):
            await app.order_target_percent("AAPL", 0.25)
    
    @pytest.mark.asyncio
    async def test_order_failure_handling(self, test_config, mock_ibapi):
        """Test handling of order placement failures."""
        app = IBKRApp()
        app.config = test_config
        
        mock_order_manager = AsyncMock()
        mock_order_manager.place_order.side_effect = Exception("Order placement failed")
        
        mock_client = Mock()
        mock_client.is_connected = True
        
        app.order_manager = mock_order_manager
        app.client = mock_client
        
        result = await app.place_buy_order("AAPL", 100)
        
        assert result["status"] == "failed"
        assert "Order placement failed" in result["error"]
        assert result["symbol"] == "AAPL"


class TestApplicationIntegration:
    """Integration tests for the full application."""
    
    @pytest.mark.asyncio
    async def test_full_application_workflow(self, temp_dir, mock_ibapi):
        """Test a complete application workflow."""
        # Create test config with temporary database
        from src.core.config import IBConfig, DatabaseConfig
        
        test_config = IBConfig(
            host="127.0.0.1",
            port=7497,
            client_id=999,
            connection_timeout=5,
            request_timeout=5,
            max_reconnect_attempts=2,
            reconnect_delay=1.0,
            database=DatabaseConfig(
                path=str(temp_dir / "test_integration.sqlite"),
                pool_size=2
            ),
            risk_free_rate=0.02
        )
        
        app = IBKRApp()
        app.config = test_config
        
        with patch('src.app.main.IBClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.start.return_value = True
            mock_client.is_connected = True
            mock_client.connection_info = {'connected': True, 'uptime_seconds': 100}
            mock_client.get_positions.return_value = []
            mock_client.get_account_values.return_value = []
            mock_client_class.return_value = mock_client
            
            try:
                # Start application
                success = await app.start()
                assert success
                assert app.running
                
                # Test portfolio summary
                summary = await app.get_portfolio_summary()
                assert 'timestamp' in summary
                
                # Test status
                status = app.status
                assert status['running']
                assert status['connected']
                
            finally:
                # Clean up
                await app.stop()
                assert not app.running
    
    @pytest.mark.asyncio
    async def test_trading_workflow_integration(self, temp_dir, mock_ibapi):
        """Test complete trading workflow integration."""
        from src.core.config import IBConfig, DatabaseConfig
        
        test_config = IBConfig(
            host="127.0.0.1",
            port=7497,
            client_id=998,
            database=DatabaseConfig(
                path=str(temp_dir / "test_trading.sqlite"),
                pool_size=2
            ),
            risk_free_rate=0.02
        )
        
        app = IBKRApp()
        app.config = test_config
        
        with patch('src.app.main.IBClient') as mock_client_class, \
             patch('src.app.main.OrderManager') as mock_order_manager_class:
            
            # Mock client
            mock_client = AsyncMock()
            mock_client.start.return_value = True
            mock_client.is_connected = True
            mock_client_class.return_value = mock_client
            
            # Mock order manager
            mock_order_manager = AsyncMock()
            mock_order_manager.place_order.return_value = {"status": "placed", "order_id": 123}
            mock_order_manager.order_target_percent.return_value = {"status": "placed"}
            mock_order_manager.cancel_all_orders.return_value = {"status": "completed", "total_cancelled": 0}
            mock_order_manager_class.return_value = mock_order_manager
            
            try:
                # Start application
                success = await app.start()
                assert success
                
                # Test basic order
                result = await app.place_buy_order("AAPL", 100)
                assert result["status"] == "placed"
                
                # Test target percent order
                result = await app.order_target_percent("AAPL", 0.25)
                assert result["status"] == "placed"
                
                # Test cancel all orders
                result = await app.cancel_all_orders()
                assert result["status"] == "completed"
                
            finally:
                await app.stop()
