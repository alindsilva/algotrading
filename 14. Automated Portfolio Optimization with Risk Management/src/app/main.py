"""
Main IBKR trading application with async portfolio optimization and risk management.
Orchestrates all components including API client, data storage, analytics, and risk management.
"""

import asyncio
import logging
import signal
from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime, timedelta
from pathlib import Path
import json

from ibapi.contract import Contract

from ..core.config import IBConfig, load_config
from ..core.exceptions import ConfigurationError, ConnectionError
from ..api.client import IBClient
from ..data.storage import AsyncDataStorage
from ..contracts.factory import ContractFactory
from ..analytics.portfolio import PortfolioAnalytics
from ..orders.manager import OrderManager, OrderRequest, OrderType, OrderAction, TimeInForce
from ..core.types import MarketDataType, PositionData, AccountValue

logger = logging.getLogger(__name__)


class IBKRApp:
    """
    Main IBKR application providing async portfolio management and risk analytics.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the IBKR application
        
        Args:
            config_path: Path to configuration file (optional)
        """
        # Load configuration
        self.config = load_config(config_path) if config_path else IBConfig()
        
        # Initialize components
        self.client: Optional[IBClient] = None
        self.storage: Optional[AsyncDataStorage] = None
        self.analytics: Optional[PortfolioAnalytics] = None
        self.order_manager: Optional[OrderManager] = None
        
        # Application state
        self.running = False
        self.shutdown_event = asyncio.Event()
        
        # Active subscriptions tracking
        self.active_subscriptions: Dict[str, asyncio.Task] = {}
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        
        # Performance tracking
        self.last_portfolio_update: Optional[datetime] = None
        self.metrics_update_interval = timedelta(minutes=5)
        
    async def start(self) -> bool:
        """
        Start the application and all components
        
        Returns:
            True if started successfully, False otherwise
        """
        if self.running:
            logger.warning("Application is already running")
            return True
        
        try:
            logger.info("Starting IBKR trading application...")
            
            # Initialize components
            await self._initialize_components()
            
            # Start the IBKR client
            if not self.client:
                logger.error("IBKR client not initialized")
                return False
            if not await self.client.start():
                logger.error("Failed to start IBKR client")
                return False

            # Start background tasks
            await self._start_background_tasks()

            self.running = True
            logger.info("IBKR application started successfully")

            return True
            
        except Exception as e:
            logger.error(f"Failed to start application: {e}")
            await self.stop()
            return False
    
    async def stop(self):
        """Stop the application and cleanup resources"""
        if not self.running:
            return
        
        logger.info("Stopping IBKR application...")
        self.running = False
        self.shutdown_event.set()
        
        # Stop background tasks
        await self._stop_background_tasks()
        
        # Stop active subscriptions
        await self._stop_all_subscriptions()
        
        # Stop components
        if self.client:
            await self.client.stop()
        
        if self.storage:
            await self.storage.close()
        
        logger.info("IBKR application stopped")
    
    async def _initialize_components(self):
        """Initialize all application components"""
        # Initialize IBKR client
        self.client = IBClient(self.config)
        
        # Initialize data storage
        self.storage = AsyncDataStorage(self.config.database)
        await self.storage.initialize()
        
        # Initialize analytics
        self.analytics = PortfolioAnalytics(
            risk_free_rate=self.config.risk_free_rate
        )
        
        # Initialize order manager
        self.order_manager = OrderManager(self.client)
        
        logger.info("All components initialized")
    
    async def _start_background_tasks(self):
        """Start background monitoring and update tasks"""
        # Portfolio metrics update task
        self.background_tasks.append(
            asyncio.create_task(self._portfolio_metrics_loop())
        )
        
        # Connection health monitoring
        self.background_tasks.append(
            asyncio.create_task(self._connection_health_loop())
        )
        
        # Data cleanup task
        self.background_tasks.append(
            asyncio.create_task(self._data_cleanup_loop())
        )
        
        logger.info(f"Started {len(self.background_tasks)} background tasks")
    
    async def _stop_background_tasks(self):
        """Stop all background tasks"""
        for task in self.background_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self.background_tasks.clear()
        logger.info("All background tasks stopped")
    
    async def _stop_all_subscriptions(self):
        """Stop all active market data subscriptions"""
        for symbol, task in self.active_subscriptions.items():
            if not task.done():
                logger.info(f"Stopping subscription for {symbol}")
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self.active_subscriptions.clear()
        logger.info("All subscriptions stopped")
    
    async def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive portfolio summary
        
        Returns:
            Dictionary containing portfolio metrics and positions
        """
        if not self.client or not self.client.is_connected:
            raise ConnectionError("Not connected to IBKR")
        
        # Get current positions
        positions = await self.client.get_positions()
        
        # Get account values
        account_values = await self.client.get_account_values()
        
        # Analyze positions
        position_analysis = await self.analytics.analyze_positions(positions, account_values)
        
        # Get historical returns for metrics calculation
        returns_data = await self.storage.get_portfolio_returns(days=252)
        portfolio_metrics = None
        
        if returns_data is not None and len(returns_data) > 5:
            portfolio_metrics = await self.analytics.calculate_portfolio_metrics(returns_data)
        
        return {
            'timestamp': datetime.now(),
            'connection_status': self.client.connection_info,
            'positions': position_analysis,
            'portfolio_metrics': {
                'total_return': portfolio_metrics.total_return if portfolio_metrics else None,
                'annualized_return': portfolio_metrics.annualized_return if portfolio_metrics else None,
                'volatility': portfolio_metrics.volatility if portfolio_metrics else None,
                'sharpe_ratio': portfolio_metrics.sharpe_ratio if portfolio_metrics else None,
                'max_drawdown': portfolio_metrics.max_drawdown if portfolio_metrics else None,
                'var_95': portfolio_metrics.var_95 if portfolio_metrics else None,
                'cvar_95': portfolio_metrics.cvar_95 if portfolio_metrics else None,
            } if portfolio_metrics else None,
            'account_summary': {
                'net_liquidation': self._extract_account_value(account_values, 'NetLiquidation'),
                'total_cash': self._extract_account_value(account_values, 'TotalCashValue'),
                'buying_power': self._extract_account_value(account_values, 'BuyingPower'),
                'gross_position_value': self._extract_account_value(account_values, 'GrossPositionValue'),
            }
        }
    
    def _extract_account_value(self, account_values: List[AccountValue], key: str) -> Optional[float]:
        """Extract specific account value by key"""
        for value in account_values:
            if value.key == key and value.currency == 'USD':
                try:
                    return float(value.value)
                except (ValueError, TypeError):
                    return None
        return None
    
    async def start_market_data_stream(self, symbol: str, 
                                     security_type: str = "STK", 
                                     exchange: str = "SMART") -> bool:
        """
        Start streaming market data for a symbol
        
        Args:
            symbol: Stock symbol
            security_type: Security type (STK, OPT, FUT, etc.)
            exchange: Exchange
        
        Returns:
            True if streaming started successfully
        """
        if not self.client or not self.client.is_connected:
            raise ConnectionError("Not connected to IBKR")
        
        if symbol in self.active_subscriptions:
            logger.warning(f"Market data subscription already active for {symbol}")
            return True
        
        try:
            # Create contract
            contract = ContractFactory.create_stock(symbol, exchange)
            
            # Start streaming task
            stream_task = asyncio.create_task(
                self._market_data_stream_handler(symbol, contract)
            )
            
            self.active_subscriptions[symbol] = stream_task
            logger.info(f"Started market data streaming for {symbol}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start market data streaming for {symbol}: {e}")
            return False
    
    async def stop_market_data_stream(self, symbol: str) -> bool:
        """
        Stop streaming market data for a symbol
        
        Args:
            symbol: Stock symbol
        
        Returns:
            True if stopped successfully
        """
        if symbol not in self.active_subscriptions:
            logger.warning(f"No active subscription found for {symbol}")
            return True
        
        task = self.active_subscriptions.pop(symbol)
        task.cancel()
        
        try:
            await task
        except asyncio.CancelledError:
            pass
        
        logger.info(f"Stopped market data streaming for {symbol}")
        return True
    
    async def _market_data_stream_handler(self, symbol: str, contract: Contract):
        """Handle streaming market data for a symbol"""
        try:
            async for tick_data in self.client.get_streaming_data(contract):
                # Store tick data
                await self.storage.store_market_data(
                    symbol=symbol,
                    timestamp=tick_data['timestamp'],
                    price=tick_data.get('price'),
                    size=tick_data.get('size'),
                    tick_type=tick_data.get('tick_type')
                )
                
                # Check for shutdown
                if not self.running:
                    break
                    
        except asyncio.CancelledError:
            logger.info(f"Market data streaming cancelled for {symbol}")
        except Exception as e:
            logger.error(f"Error in market data streaming for {symbol}: {e}")
            # Remove from active subscriptions on error
            if symbol in self.active_subscriptions:
                del self.active_subscriptions[symbol]
    
    async def get_risk_report(self, days: int = 30) -> Dict[str, Any]:
        """
        Generate comprehensive risk report
        
        Args:
            days: Number of days of historical data to analyze
        
        Returns:
            Risk report dictionary
        """
        if not self.analytics:
            raise RuntimeError("Analytics not initialized")
        
        # Get portfolio returns
        returns = await self.storage.get_portfolio_returns(days=days)
        if returns is None or len(returns) < 5:
            return {
                'error': 'Insufficient historical data for risk analysis',
                'days_requested': days,
                'data_points_available': len(returns) if returns is not None else 0
            }
        
        # Calculate portfolio metrics
        metrics = await self.analytics.calculate_portfolio_metrics(returns)
        
        # Calculate rolling metrics
        rolling_metrics = await self.analytics.calculate_rolling_metrics(returns, window=min(21, len(returns)//2))
        
        # Perform stress testing
        stress_results = await self.analytics.stress_test_portfolio(returns)
        
        # Get current positions
        positions = await self.client.get_positions()
        account_values = await self.client.get_account_values()
        position_analysis = await self.analytics.analyze_positions(positions, account_values)
        
        return {
            'timestamp': datetime.now(),
            'analysis_period_days': days,
            'portfolio_metrics': {
                'total_return': metrics.total_return,
                'annualized_return': metrics.annualized_return,
                'volatility': metrics.volatility,
                'max_drawdown': metrics.max_drawdown,
                'sharpe_ratio': metrics.sharpe_ratio,
                'sortino_ratio': metrics.sortino_ratio,
                'calmar_ratio': metrics.calmar_ratio,
                'omega_ratio': metrics.omega_ratio,
                'var_95': metrics.var_95,
                'cvar_95': metrics.cvar_95,
            },
            'rolling_metrics': {
                'latest_volatility': rolling_metrics['rolling_volatility'].iloc[-1] if len(rolling_metrics['rolling_volatility']) > 0 else None,
                'latest_sharpe': rolling_metrics['rolling_sharpe_ratio'].iloc[-1] if len(rolling_metrics['rolling_sharpe_ratio']) > 0 else None,
                'volatility_trend': self._determine_volatility_trend(rolling_metrics['rolling_volatility'])
            },
            'stress_test_results': stress_results,
            'position_analysis': position_analysis,
            'risk_alerts': self._generate_risk_alerts(metrics, position_analysis)
        }
    
    def _generate_risk_alerts(self, metrics, position_analysis) -> List[Dict[str, str]]:
        """Generate risk alerts based on metrics and positions"""
        alerts = []
        
        # Volatility alert
        if metrics.volatility > 0.25:  # 25% annualized volatility
            alerts.append({
                'type': 'HIGH_VOLATILITY',
                'severity': 'WARNING',
                'message': f'High portfolio volatility: {metrics.volatility:.2%}'
            })
        
        # Drawdown alert
        if metrics.max_drawdown > 0.10:  # 10% max drawdown
            alerts.append({
                'type': 'HIGH_DRAWDOWN',
                'severity': 'WARNING',
                'message': f'High maximum drawdown: {metrics.max_drawdown:.2%}'
            })
        
        # Concentration alert
        for position in position_analysis['largest_positions'][:3]:  # Top 3 positions
            if position['pct_of_portfolio'] > 0.20:  # 20% concentration
                alerts.append({
                    'type': 'CONCENTRATION_RISK',
                    'severity': 'INFO',
                    'message': f'High concentration in {position["symbol"]}: {position["pct_of_portfolio"]:.1%}'
                })
        
        # Low Sharpe ratio alert
        if metrics.sharpe_ratio < 0.5:
            alerts.append({
                'type': 'LOW_SHARPE',
                'severity': 'INFO',
                'message': f'Low risk-adjusted returns (Sharpe ratio: {metrics.sharpe_ratio:.2f})'
            })
        
        return alerts
    
    def _determine_volatility_trend(self, rolling_volatility) -> str:
        """Determine volatility trend from rolling volatility series"""
        if len(rolling_volatility) < 2:
            return 'insufficient_data'
        
        # Compare latest with 5 periods back, or as far back as we have data
        lookback = min(5, len(rolling_volatility) - 1)
        if rolling_volatility.iloc[-1] > rolling_volatility.iloc[-lookback - 1]:
            return 'increasing'
        else:
            return 'decreasing'
    
    async def _portfolio_metrics_loop(self):
        """Background task to periodically update portfolio metrics"""
        while self.running:
            try:
                if (self.last_portfolio_update is None or 
                    datetime.now() - self.last_portfolio_update > self.metrics_update_interval):
                    
                    # Update portfolio metrics
                    await self._update_portfolio_metrics()
                    self.last_portfolio_update = datetime.now()
                    
                    logger.debug("Portfolio metrics updated")
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in portfolio metrics loop: {e}")
                await asyncio.sleep(60)
    
    async def _update_portfolio_metrics(self):
        """Update and store current portfolio metrics"""
        try:
            if not self.client or not self.client.is_connected:
                return
            
            # Get current positions and account values
            positions = await self.client.get_positions()
            account_values = await self.client.get_account_values()
            
            # Store current portfolio snapshot
            net_liquidation = self._extract_account_value(account_values, 'NetLiquidation')
            if net_liquidation:
                await self.storage.store_portfolio_value(
                    timestamp=datetime.now(),
                    total_value=net_liquidation,
                    positions=positions
                )
            
            # Update last update time
            self.last_portfolio_update = datetime.now()
                
        except Exception as e:
            logger.error(f"Failed to update portfolio metrics: {e}")
    
    async def _connection_health_loop(self):
        """Background task to monitor connection health"""
        while self.running:
            try:
                if self.client:
                    connection_info = self.client.connection_info
                    
                    if not connection_info['connected']:
                        logger.warning("Connection lost - attempting reconnection")
                        # Connection manager will handle reconnection automatically
                    
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in connection health loop: {e}")
                await asyncio.sleep(30)
    
    async def _data_cleanup_loop(self):
        """Background task to perform data cleanup"""
        while self.running:
            try:
                # Perform cleanup once per day
                await asyncio.sleep(86400)  # 24 hours
                
                if self.storage:
                    await self.storage.cleanup_old_data(days_to_keep=90)
                    logger.info("Performed daily data cleanup")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in data cleanup loop: {e}")
    
    async def run_until_signal(self):
        """Run the application until a shutdown signal is received"""
        def signal_handler(sig, frame):
            logger.info(f"Received signal {sig}, shutting down...")
            self.shutdown_event.set()
        
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Wait for shutdown signal
        await self.shutdown_event.wait()
    
    # Trading methods using OrderManager
    
    async def place_buy_order(self, symbol: str, quantity: int, 
                             order_type: str = "MKT", 
                             limit_price: Optional[float] = None,
                             stop_price: Optional[float] = None,
                             time_in_force: str = "DAY",
                             dry_run: bool = False) -> Dict[str, Any]:
        """
        Place a buy order through the order manager
        
        Args:
            symbol: Stock symbol to buy
            quantity: Number of shares to buy
            order_type: Order type (MKT, LMT, STP, etc.)
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            time_in_force: Time in force (DAY, GTC, IOC, FOK)
            dry_run: If True, validate but don't place the order
            
        Returns:
            Order result dictionary
        """
        if not self.order_manager:
            raise RuntimeError("Order manager not initialized")
        
        if not self.client or not self.client.is_connected:
            raise ConnectionError("Not connected to IBKR")
        
        try:
            # Create order request
            order_request = OrderRequest(
                symbol=symbol,
                action=OrderAction.BUY,
                quantity=quantity,
                order_type=OrderType(order_type),
                limit_price=limit_price,
                stop_price=stop_price,
                time_in_force=TimeInForce(time_in_force)
            )
            
            # Place order through order manager
            result = await self.order_manager.place_order(order_request, dry_run=dry_run)
            
            logger.info(f"Buy order {'validated' if dry_run else 'placed'}: {symbol} x{quantity} @ {order_type}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to place buy order for {symbol}: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "symbol": symbol,
                "action": "BUY",
                "quantity": quantity
            }
    
    async def place_sell_order(self, symbol: str, quantity: int,
                              order_type: str = "MKT",
                              limit_price: Optional[float] = None,
                              stop_price: Optional[float] = None,
                              time_in_force: str = "DAY",
                              dry_run: bool = False) -> Dict[str, Any]:
        """
        Place a sell order through the order manager
        
        Args:
            symbol: Stock symbol to sell
            quantity: Number of shares to sell
            order_type: Order type (MKT, LMT, STP, etc.)
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            time_in_force: Time in force (DAY, GTC, IOC, FOK)
            dry_run: If True, validate but don't place the order
            
        Returns:
            Order result dictionary
        """
        if not self.order_manager:
            raise RuntimeError("Order manager not initialized")
        
        if not self.client or not self.client.is_connected:
            raise ConnectionError("Not connected to IBKR")
        
        try:
            # Create order request
            order_request = OrderRequest(
                symbol=symbol,
                action=OrderAction.SELL,
                quantity=quantity,
                order_type=OrderType(order_type),
                limit_price=limit_price,
                stop_price=stop_price,
                time_in_force=TimeInForce(time_in_force)
            )
            
            # Place order through order manager
            result = await self.order_manager.place_order(order_request, dry_run=dry_run)
            
            logger.info(f"Sell order {'validated' if dry_run else 'placed'}: {symbol} x{quantity} @ {order_type}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to place sell order for {symbol}: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "symbol": symbol,
                "action": "SELL",
                "quantity": quantity
            }
    
    async def cancel_order(self, order_id: int) -> Dict[str, Any]:
        """
        Cancel an active order
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            Cancellation result dictionary
        """
        if not self.order_manager:
            raise RuntimeError("Order manager not initialized")
        
        return await self.order_manager.cancel_order(order_id)
    
    async def get_order_status(self, order_id: int) -> Optional[Dict[str, Any]]:
        """
        Get status of a specific order
        
        Args:
            order_id: Order ID to query
            
        Returns:
            Order status dictionary or None if not found
        """
        if not self.order_manager:
            raise RuntimeError("Order manager not initialized")
        
        return await self.order_manager.get_order_status(order_id)
    
    def get_active_orders(self) -> List[Dict[str, Any]]:
        """
        Get all currently active orders
        
        Returns:
            List of active order dictionaries
        """
        if not self.order_manager:
            raise RuntimeError("Order manager not initialized")
        
        return self.order_manager.get_active_orders()
    
    def get_order_history(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get order history for specified number of days
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of historical order dictionaries
        """
        if not self.order_manager:
            raise RuntimeError("Order manager not initialized")
        
        return self.order_manager.get_order_history(days=days)
    
    async def get_market_quote(self, symbol: str, 
                              security_type: str = "STK", 
                              exchange: str = "SMART") -> Optional[Dict[str, Any]]:
        """
        Get current market quote for a symbol
        
        Args:
            symbol: Stock symbol
            security_type: Security type (STK, OPT, FUT, etc.)
            exchange: Exchange
            
        Returns:
            Market quote dictionary or None if failed
        """
        if not self.client or not self.client.is_connected:
            raise ConnectionError("Not connected to IBKR")
        
        try:
            # Create contract
            contract = ContractFactory.create_stock(symbol, exchange)
            
            # Get market data
            market_data = await self.client.get_market_data(
                contract=contract,
                timeout=10.0
            )
            
            if market_data:
                return {
                    'symbol': symbol,
                    'timestamp': datetime.now(),
                    'bid': market_data.get('bid'),
                    'ask': market_data.get('ask'),
                    'last': market_data.get('last'),
                    'volume': market_data.get('volume'),
                    'high': market_data.get('high'),
                    'low': market_data.get('low'),
                    'close': market_data.get('close')
                }
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to get market quote for {symbol}: {e}")
            return None
    
    async def close_position(self, symbol: str, dry_run: bool = False) -> Dict[str, Any]:
        """
        Close entire position for a symbol (sell all shares)
        
        Args:
            symbol: Stock symbol to close position for
            dry_run: If True, validate but don't place the order
            
        Returns:
            Order result dictionary
        """
        try:
            # Get current position
            positions = await self.client.get_positions()
            current_position = 0
            
            for position in positions:
                if position.symbol == symbol:
                    current_position = int(position.position)
                    break
            
            if current_position == 0:
                return {
                    "status": "no_position",
                    "message": f"No position found for {symbol}",
                    "symbol": symbol
                }
            
            # Determine action and quantity
            if current_position > 0:
                # Long position - sell to close
                action = "SELL"
                quantity = current_position
            else:
                # Short position - buy to cover
                action = "BUY"
                quantity = abs(current_position)
            
            # Place market order to close position
            if action == "BUY":
                result = await self.place_buy_order(
                    symbol=symbol,
                    quantity=quantity,
                    order_type="MKT",
                    dry_run=dry_run
                )
            else:
                result = await self.place_sell_order(
                    symbol=symbol,
                    quantity=quantity,
                    order_type="MKT",
                    dry_run=dry_run
                )
            
            result['close_position'] = True
            result['original_position'] = current_position
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to close position for {symbol}: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "symbol": symbol
            }
    
    async def order_target_percent(self, symbol: str, target_percent: float,
                                  order_type: str = "MKT",
                                  limit_price: Optional[float] = None,
                                  stop_price: Optional[float] = None,
                                  time_in_force: str = "DAY",
                                  dry_run: bool = False) -> Dict[str, Any]:
        """
        Place an order to adjust a position to a target percent of portfolio value.
        
        This is the key portfolio management function that automatically calculates
        whether to buy or sell to reach the target allocation percentage.
        
        Args:
            symbol: Stock symbol
            target_percent: Target allocation as decimal (0.1 = 10% of portfolio)
            order_type: Order type (MARKET, LIMIT, STOP, etc.)
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            time_in_force: Time in force (DAY, GTC, IOC, FOK)
            dry_run: If True, validate but don't place the order
            
        Returns:
            Order result dictionary with portfolio context
        """
        if not self.order_manager:
            raise RuntimeError("Order manager not initialized")
        
        if not self.client or not self.client.is_connected:
            raise ConnectionError("Not connected to IBKR")
        
        try:
            # Use OrderManager's implementation
            result = await self.order_manager.order_target_percent(
                symbol=symbol,
                target_percent=target_percent,
                order_type=OrderType(order_type),
                limit_price=limit_price,
                stop_price=stop_price,
                time_in_force=TimeInForce(time_in_force),
                dry_run=dry_run
            )
            
            action = "TARGET" if result.get('status') in ['placed', 'validated'] else "CHECK"
            logger.info(f"{action} {symbol} to {target_percent:.1%} allocation: {result.get('status')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to place target percent order for {symbol}: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "symbol": symbol,
                "target_percent": target_percent
            }
    
    async def order_percent(self, symbol: str, percent: float,
                           order_type: str = "MKT",
                           limit_price: Optional[float] = None,
                           stop_price: Optional[float] = None,
                           time_in_force: str = "DAY",
                           dry_run: bool = False) -> Dict[str, Any]:
        """
        Place an order for a percentage of current portfolio value.
        
        This places a new order without considering existing positions.
        Use order_target_percent for portfolio allocation management.
        
        Args:
            symbol: Stock symbol  
            percent: Percentage of portfolio as decimal (0.1 = 10% of portfolio)
            order_type: Order type (MARKET, LIMIT, etc.)
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            time_in_force: Time in force (DAY, GTC, IOC, FOK)
            dry_run: If True, validate but don't place the order
            
        Returns:
            Order result dictionary
        """
        if not self.order_manager:
            raise RuntimeError("Order manager not initialized")
        
        if not self.client or not self.client.is_connected:
            raise ConnectionError("Not connected to IBKR")
        
        try:
            result = await self.order_manager.order_percent(
                symbol=symbol,
                percent=percent,
                order_type=OrderType(order_type),
                limit_price=limit_price,
                stop_price=stop_price,
                time_in_force=TimeInForce(time_in_force),
                dry_run=dry_run
            )
            
            action = "BUY" if percent > 0 else "SELL"
            logger.info(f"{action} {symbol} for {abs(percent):.1%} of portfolio: {result.get('status')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to place percent order for {symbol}: {e}")
            return {
                "status": "failed", 
                "error": str(e),
                "symbol": symbol,
                "percent": percent
            }
    
    async def rebalance_portfolio(self, target_allocations: Dict[str, float],
                                 order_type: str = "MKT",
                                 dry_run: bool = False) -> Dict[str, Any]:
        """
        Rebalance entire portfolio to target allocations.
        
        This is a powerful function that can rebalance your entire portfolio
        in one operation, automatically calculating all required trades.
        
        Args:
            target_allocations: Dict of {symbol: target_percent} 
                              e.g. {"AAPL": 0.25, "MSFT": 0.20, "GOOGL": 0.15}
            order_type: Order type to use for all orders (MARKET, LIMIT, etc.)
            dry_run: If True, validate but don't place orders
            
        Returns:
            Dict with results for each symbol and summary statistics
        """
        if not self.order_manager:
            raise RuntimeError("Order manager not initialized")
        
        if not self.client or not self.client.is_connected:
            raise ConnectionError("Not connected to IBKR")
        
        try:
            result = await self.order_manager.rebalance_portfolio(
                target_allocations=target_allocations,
                order_type=OrderType(order_type),
                dry_run=dry_run
            )
            
            summary = result.get('summary', {})
            total_symbols = summary.get('total_symbols', 0)
            orders_placed = summary.get('orders_placed', 0)
            
            logger.info(f"Portfolio rebalance {'validated' if dry_run else 'completed'}: "
                       f"{orders_placed}/{total_symbols} orders processed")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to rebalance portfolio: {e}")
            return {
                "summary": {
                    "status": "failed",
                    "error": str(e),
                    "dry_run": dry_run
                },
                "results": {}
            }
    
    async def get_portfolio_allocations(self) -> Dict[str, float]:
        """
        Get current portfolio allocations as percentages.
        
        Returns:
            Dict of {symbol: current_percent} for all positions
        """
        if not self.client or not self.client.is_connected:
            raise ConnectionError("Not connected to IBKR")
        
        try:
            # Get positions and account values
            positions = await self.client.get_positions()
            account_values = await self.client.get_account_values()
            
            # Get net liquidation value
            net_liquidation = 0
            for value in account_values:
                if value.key == 'NetLiquidation' and value.currency == 'USD':
                    net_liquidation = float(value.value)
                    break
            
            if net_liquidation <= 0:
                return {}
            
            # Calculate allocations
            allocations = {}
            
            for position in positions:
                if position.position != 0:  # Only include non-zero positions
                    # Get current market price
                    try:
                        quote = await self.get_market_quote(position.symbol)
                        if quote and quote.get('last'):
                            current_price = float(quote['last'])
                            position_value = position.position * current_price
                            allocation = position_value / net_liquidation
                            allocations[position.symbol] = allocation
                    except Exception as e:
                        logger.warning(f"Could not get price for {position.symbol}: {e}")
            
            return allocations
            
        except Exception as e:
            logger.error(f"Failed to get portfolio allocations: {e}")
            return {}
    
    async def suggest_rebalance(self, target_allocations: Dict[str, float],
                               threshold: float = 0.05) -> Dict[str, Any]:
        """
        Suggest rebalancing trades based on current vs target allocations.
        
        Args:
            target_allocations: Dict of {symbol: target_percent}
            threshold: Minimum deviation to trigger rebalance suggestion (default 5%)
            
        Returns:
            Dict with suggested trades and analysis
        """
        try:
            # Get current allocations
            current_allocations = await self.get_portfolio_allocations()
            
            suggestions = {}
            needs_rebalance = False
            
            # Check each target allocation
            for symbol, target_percent in target_allocations.items():
                current_percent = current_allocations.get(symbol, 0.0)
                deviation = abs(target_percent - current_percent)
                
                if deviation >= threshold:
                    needs_rebalance = True
                    action = "INCREASE" if target_percent > current_percent else "DECREASE"
                    
                    suggestions[symbol] = {
                        "action": action,
                        "current_percent": current_percent,
                        "target_percent": target_percent,
                        "deviation": deviation,
                        "deviation_pct": deviation / max(target_percent, 0.01) * 100
                    }
            
            # Check for positions not in target (should be reduced/eliminated)
            for symbol, current_percent in current_allocations.items():
                if symbol not in target_allocations and current_percent >= threshold:
                    needs_rebalance = True
                    suggestions[symbol] = {
                        "action": "ELIMINATE",
                        "current_percent": current_percent,
                        "target_percent": 0.0,
                        "deviation": current_percent,
                        "deviation_pct": 100.0
                    }
            
            return {
                "needs_rebalance": needs_rebalance,
                "threshold_used": threshold,
                "suggestions": suggestions,
                "current_allocations": current_allocations,
                "target_allocations": target_allocations,
                "total_target_allocation": sum(target_allocations.values())
            }
            
        except Exception as e:
            logger.error(f"Failed to generate rebalance suggestions: {e}")
            return {
                "needs_rebalance": False,
                "error": str(e)
            }
    
    async def order_target_quantity(self, symbol: str, target_quantity: int,
                                   order_type: str = "MKT",
                                   limit_price: Optional[float] = None,
                                   stop_price: Optional[float] = None,
                                   time_in_force: str = "DAY",
                                   dry_run: bool = False) -> Dict[str, Any]:
        """
        Place an order to adjust a position to a target number of shares.
        
        Args:
            symbol: Stock symbol
            target_quantity: Target number of shares (can be negative for short)
            order_type: Order type (MARKET, LIMIT, STOP, etc.)
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            time_in_force: Time in force (DAY, GTC, IOC, FOK)
            dry_run: If True, validate but don't place the order
            
        Returns:
            Order result dictionary
        """
        if not self.order_manager:
            raise RuntimeError("Order manager not initialized")
        
        if not self.client or not self.client.is_connected:
            raise ConnectionError("Not connected to IBKR")
        
        try:
            result = await self.order_manager.order_target_quantity(
                symbol=symbol,
                target_quantity=target_quantity,
                order_type=OrderType(order_type),
                limit_price=limit_price,
                stop_price=stop_price,
                time_in_force=TimeInForce(time_in_force),
                dry_run=dry_run
            )
            
            logger.info(f"Target quantity order for {symbol}: targeting {target_quantity} shares")
            return result
            
        except Exception as e:
            logger.error(f"Failed to place target quantity order for {symbol}: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "symbol": symbol,
                "target_quantity": target_quantity
            }
    
    async def order_target_value(self, symbol: str, target_value: float,
                                order_type: str = "MKT",
                                limit_price: Optional[float] = None,
                                stop_price: Optional[float] = None,
                                time_in_force: str = "DAY",
                                dry_run: bool = False) -> Dict[str, Any]:
        """
        Place an order to adjust a position to a target dollar value.
        
        Args:
            symbol: Stock symbol
            target_value: Target position value in dollars
            order_type: Order type (MARKET, LIMIT, STOP, etc.)
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            time_in_force: Time in force (DAY, GTC, IOC, FOK)
            dry_run: If True, validate but don't place the order
            
        Returns:
            Order result dictionary
        """
        if not self.order_manager:
            raise RuntimeError("Order manager not initialized")
        
        if not self.client or not self.client.is_connected:
            raise ConnectionError("Not connected to IBKR")
        
        try:
            result = await self.order_manager.order_target_value(
                symbol=symbol,
                target_value=target_value,
                order_type=OrderType(order_type),
                limit_price=limit_price,
                stop_price=stop_price,
                time_in_force=TimeInForce(time_in_force),
                dry_run=dry_run
            )
            
            logger.info(f"Target value order for {symbol}: targeting ${target_value:,.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to place target value order for {symbol}: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "symbol": symbol,
                "target_value": target_value
            }
    
    async def order_value(self, symbol: str, value: float,
                         order_type: str = "MKT",
                         limit_price: Optional[float] = None,
                         stop_price: Optional[float] = None,
                         time_in_force: str = "DAY",
                         dry_run: bool = False) -> Dict[str, Any]:
        """
        Place an order for a fixed dollar amount.
        
        Args:
            symbol: Stock symbol
            value: Dollar amount to trade (positive for buy, negative for sell)
            order_type: Order type (MARKET, LIMIT, STOP, etc.)
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            time_in_force: Time in force (DAY, GTC, IOC, FOK)
            dry_run: If True, validate but don't place the order
            
        Returns:
            Order result dictionary
        """
        if not self.order_manager:
            raise RuntimeError("Order manager not initialized")
        
        if not self.client or not self.client.is_connected:
            raise ConnectionError("Not connected to IBKR")
        
        try:
            result = await self.order_manager.order_value(
                symbol=symbol,
                value=value,
                order_type=OrderType(order_type),
                limit_price=limit_price,
                stop_price=stop_price,
                time_in_force=TimeInForce(time_in_force),
                dry_run=dry_run
            )
            
            action = "BUY" if value > 0 else "SELL"
            logger.info(f"Value order for {symbol}: {action} ${abs(value):,.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to place value order for {symbol}: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "symbol": symbol,
                "value": value
            }
    
    async def cancel_all_orders(self) -> Dict[str, Any]:
        """
        Cancel all active orders.
        
        Returns:
            Dict with cancellation results
        """
        if not self.order_manager:
            raise RuntimeError("Order manager not initialized")
        
        if not self.client or not self.client.is_connected:
            raise ConnectionError("Not connected to IBKR")
        
        try:
            result = await self.order_manager.cancel_all_orders()
            total_cancelled = result.get('total_cancelled', 0)
            
            logger.info(f"Cancel all orders completed: {total_cancelled} orders cancelled")
            return result
            
        except Exception as e:
            logger.error(f"Failed to cancel all orders: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def cancel_order_by_id(self, order_id: int) -> Dict[str, Any]:
        """
        Cancel a specific order by ID (synchronous).
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            Cancellation result dictionary
        """
        if not self.client or not self.client.is_connected:
            raise ConnectionError("Not connected to IBKR")
        
        try:
            self.client.cancel_order_by_id(order_id)
            
            logger.info(f"Requested cancellation of order {order_id}")
            return {
                "status": "requested",
                "order_id": order_id,
                "message": f"Cancellation requested for order {order_id}"
            }
            
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "order_id": order_id
            }
    
    async def update_order(self, order_id: int, symbol: str, quantity: int,
                          order_type: str = "MKT",
                          limit_price: Optional[float] = None,
                          stop_price: Optional[float] = None,
                          time_in_force: str = "DAY",
                          dry_run: bool = False) -> Dict[str, Any]:
        """
        Update an existing order by cancelling and replacing it.
        
        Args:
            order_id: Existing order ID to update
            symbol: Stock symbol
            quantity: Number of shares
            order_type: Order type (MARKET, LIMIT, STOP, etc.)
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            time_in_force: Time in force (DAY, GTC, IOC, FOK)
            dry_run: If True, validate but don't actually update
            
        Returns:
            Update result dictionary
        """
        if not self.order_manager:
            raise RuntimeError("Order manager not initialized")
        
        if not self.client or not self.client.is_connected:
            raise ConnectionError("Not connected to IBKR")
        
        try:
            # Create new order request
            order_request = OrderRequest(
                symbol=symbol,
                action=OrderAction.BUY,  # Will be determined by quantity sign if needed
                quantity=quantity,
                order_type=OrderType(order_type),
                limit_price=limit_price,
                stop_price=stop_price,
                time_in_force=TimeInForce(time_in_force)
            )
            
            result = await self.order_manager.update_order(
                order_id=order_id,
                order_request=order_request,
                dry_run=dry_run
            )
            
            logger.info(f"Update order {order_id}: {result.get('status')}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to update order {order_id}: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "order_id": order_id
            }
    
    async def get_pnl(self, account: str = "") -> Dict[str, Any]:
        """
        Get current P&L information.
        
        Args:
            account: Account name (if empty, uses default account)
            
        Returns:
            Dict with P&L data
        """
        if not self.client or not self.client.is_connected:
            raise ConnectionError("Not connected to IBKR")
        
        try:
            pnl_data = await self.client.get_pnl(account=account)
            
            logger.info(f"Retrieved P&L data for account: {account or 'default'}")
            return pnl_data
            
        except Exception as e:
            logger.error(f"Failed to get P&L data: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "account": account
            }
    
    async def get_historical_data(self, symbol: str, duration: str, bar_size: str,
                                 what_to_show: str = "TRADES",
                                 security_type: str = "STK",
                                 exchange: str = "SMART") -> List[Dict[str, Any]]:
        """
        Get historical data for a symbol.
        
        Args:
            symbol: Stock symbol
            duration: Duration string (e.g., "1 D", "1 W", "1 M")
            bar_size: Bar size (e.g., "1 min", "5 mins", "1 day")
            what_to_show: What data to show (TRADES, MIDPOINT, BID, ASK)
            security_type: Security type (STK, OPT, FUT, etc.)
            exchange: Exchange
            
        Returns:
            List of historical bars
        """
        if not self.client or not self.client.is_connected:
            raise ConnectionError("Not connected to IBKR")
        
        try:
            # Create contract
            contract = ContractFactory.create_stock(symbol, exchange)
            
            # Get historical data
            historical_data = await self.client.get_historical_data(
                contract=contract,
                duration=duration,
                bar_size=bar_size,
                what_to_show=what_to_show
            )
            
            logger.info(f"Retrieved {len(historical_data)} bars of historical data for {symbol}")
            return historical_data
            
        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            return []
    
    @property
    def status(self) -> Dict[str, Any]:
        """Get application status"""
        return {
            'running': self.running,
            'connected': self.client.is_connected if self.client else False,
            'active_subscriptions': list(self.active_subscriptions.keys()),
            'background_tasks': len(self.background_tasks),
            'last_portfolio_update': self.last_portfolio_update.isoformat() if self.last_portfolio_update else None,
            'active_orders': len(self.order_manager.get_active_orders()) if self.order_manager else 0,
        }
