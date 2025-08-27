"""
Order management system for IBKR trading operations.
Provides high-level order management, validation, and tracking.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict

from ibapi.contract import Contract
from ibapi.order import Order

from ..core.types import OrderStatus, PositionData
from ..core.exceptions import ValidationError, OrderError
from ..api.client import IBClient

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types supported by the system"""
    MARKET = "MKT"
    LIMIT = "LMT" 
    STOP = "STP"
    STOP_LIMIT = "STP LMT"
    TRAIL = "TRAIL"
    TRAIL_LIMIT = "TRAIL LIMIT"


class OrderAction(Enum):
    """Order actions"""
    BUY = "BUY"
    SELL = "SELL"


class TimeInForce(Enum):
    """Order time in force options"""
    DAY = "DAY"
    GTC = "GTC"  # Good Till Canceled
    IOC = "IOC"  # Immediate Or Cancel
    FOK = "FOK"  # Fill Or Kill


@dataclass
class OrderRequest:
    """Order request structure"""
    symbol: str
    action: OrderAction
    quantity: int
    order_type: OrderType
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    account: Optional[str] = None
    exchange: str = "SMART"
    security_type: str = "STK"
    currency: str = "USD"
    
    def validate(self) -> List[str]:
        """Validate order request parameters"""
        errors = []
        
        if not self.symbol or not self.symbol.strip():
            errors.append("Symbol is required")
        
        if self.quantity <= 0:
            errors.append("Quantity must be positive")
        
        if self.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            if self.limit_price is None or self.limit_price <= 0:
                errors.append(f"{self.order_type.value} orders require a valid limit price")
        
        if self.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
            if self.stop_price is None or self.stop_price <= 0:
                errors.append(f"{self.order_type.value} orders require a valid stop price")
        
        if self.order_type == OrderType.STOP_LIMIT:
            if self.limit_price and self.stop_price:
                if self.action == OrderAction.BUY and self.limit_price < self.stop_price:
                    errors.append("For buy stop limit orders, limit price must be >= stop price")
                elif self.action == OrderAction.SELL and self.limit_price > self.stop_price:
                    errors.append("For sell stop limit orders, limit price must be <= stop price")
        
        return errors


@dataclass
class OrderTracker:
    """Track active order status and details"""
    order_id: int
    request: OrderRequest
    ib_order: Order
    contract: Contract
    status: str = "Created"
    filled: float = 0.0
    remaining: float = 0.0
    avg_fill_price: float = 0.0
    last_fill_price: float = 0.0
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


class OrderManager:
    """High-level order management system"""
    
    def __init__(self, client: IBClient):
        self.client = client
        self.active_orders: Dict[int, OrderTracker] = {}
        self.order_history: List[OrderTracker] = []
        
        # Risk management parameters
        self.max_position_size = 10000  # Maximum position size per symbol
        self.max_order_value = 50000   # Maximum single order value
        self.daily_loss_limit = 5000   # Daily loss limit
        
        # Tracking
        self.daily_pnl = 0.0
        self.last_reset_date = datetime.now().date()
    
    async def place_order(self, order_request: OrderRequest, 
                         dry_run: bool = False) -> Dict[str, Any]:
        """
        Place an order with validation and risk checks
        
        Args:
            order_request: Order details
            dry_run: If True, validate but don't actually place the order
            
        Returns:
            Order result dictionary
        """
        try:
            # Validate order request
            validation_errors = order_request.validate()
            if validation_errors:
                raise ValidationError(f"Order validation failed: {', '.join(validation_errors)}")
            
            # Perform risk checks
            risk_errors = await self._perform_risk_checks(order_request)
            if risk_errors:
                raise OrderError(f"Risk check failed: {', '.join(risk_errors)}")
            
            if dry_run:
                return {
                    "status": "validated",
                    "message": "Order validation successful",
                    "order_request": asdict(order_request)
                }
            
            # Create IBKR contract and order objects
            contract = self._create_contract(order_request)
            ib_order = self._create_ib_order(order_request)
            
            # Place order through IBKR client
            result = await self.client.place_order(contract, ib_order, timeout=30.0)
            
            # Track the order
            order_tracker = OrderTracker(
                order_id=result["order_id"],
                request=order_request,
                ib_order=ib_order,
                contract=contract
            )
            
            self.active_orders[result["order_id"]] = order_tracker
            
            logger.info(f"Order placed successfully: {order_request.symbol} "
                       f"{order_request.action.value} {order_request.quantity} "
                       f"@ {order_request.order_type.value}")
            
            return {
                "status": "placed",
                "order_id": result["order_id"],
                "symbol": order_request.symbol,
                "action": order_request.action.value,
                "quantity": order_request.quantity,
                "order_type": order_request.order_type.value,
                "message": "Order placed successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "order_request": asdict(order_request)
            }
    
    async def cancel_order(self, order_id: int) -> Dict[str, Any]:
        """Cancel an active order"""
        try:
            if order_id not in self.active_orders:
                return {
                    "status": "error",
                    "message": f"Order {order_id} not found in active orders"
                }
            
            # Cancel through IBKR
            self.client.cancelOrder(order_id)
            
            # Update tracking
            order_tracker = self.active_orders[order_id]
            order_tracker.status = "Cancelled"
            order_tracker.updated_at = datetime.now()
            
            # Move to history
            self.order_history.append(self.active_orders.pop(order_id))
            
            logger.info(f"Order {order_id} cancelled successfully")
            
            return {
                "status": "cancelled",
                "order_id": order_id,
                "message": "Order cancelled successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def get_order_status(self, order_id: int) -> Optional[Dict[str, Any]]:
        """Get status of a specific order"""
        if order_id in self.active_orders:
            tracker = self.active_orders[order_id]
            return {
                "order_id": order_id,
                "symbol": tracker.request.symbol,
                "action": tracker.request.action.value,
                "quantity": tracker.request.quantity,
                "order_type": tracker.request.order_type.value,
                "status": tracker.status,
                "filled": tracker.filled,
                "remaining": tracker.remaining,
                "avg_fill_price": tracker.avg_fill_price,
                "created_at": tracker.created_at.isoformat(),
                "updated_at": tracker.updated_at.isoformat()
            }
        
        # Check order history
        for tracker in self.order_history:
            if tracker.order_id == order_id:
                return {
                    "order_id": order_id,
                    "symbol": tracker.request.symbol,
                    "action": tracker.request.action.value,
                    "quantity": tracker.request.quantity,
                    "order_type": tracker.request.order_type.value,
                    "status": tracker.status,
                    "filled": tracker.filled,
                    "remaining": tracker.remaining,
                    "avg_fill_price": tracker.avg_fill_price,
                    "created_at": tracker.created_at.isoformat(),
                    "updated_at": tracker.updated_at.isoformat()
                }
        
        return None
    
    def get_active_orders(self) -> List[Dict[str, Any]]:
        """Get all active orders"""
        return [
            {
                "order_id": order_id,
                "symbol": tracker.request.symbol,
                "action": tracker.request.action.value,
                "quantity": tracker.request.quantity,
                "order_type": tracker.request.order_type.value,
                "status": tracker.status,
                "filled": tracker.filled,
                "remaining": tracker.remaining,
                "created_at": tracker.created_at.isoformat(),
                "updated_at": tracker.updated_at.isoformat()
            }
            for order_id, tracker in self.active_orders.items()
        ]
    
    def get_order_history(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get order history for specified number of days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        return [
            {
                "order_id": tracker.order_id,
                "symbol": tracker.request.symbol,
                "action": tracker.request.action.value,
                "quantity": tracker.request.quantity,
                "order_type": tracker.request.order_type.value,
                "status": tracker.status,
                "filled": tracker.filled,
                "remaining": tracker.remaining,
                "avg_fill_price": tracker.avg_fill_price,
                "created_at": tracker.created_at.isoformat(),
                "updated_at": tracker.updated_at.isoformat()
            }
            for tracker in self.order_history
            if tracker.created_at >= cutoff_date
        ]
    
    async def _perform_risk_checks(self, order_request: OrderRequest) -> List[str]:
        """Perform risk management checks"""
        errors = []
        
        # Reset daily tracking if new day
        current_date = datetime.now().date()
        if current_date != self.last_reset_date:
            self.daily_pnl = 0.0
            self.last_reset_date = current_date
        
        # Check daily loss limit
        if self.daily_pnl < -self.daily_loss_limit:
            errors.append(f"Daily loss limit exceeded: ${abs(self.daily_pnl):,.2f}")
        
        # Estimate order value for validation
        estimated_price = order_request.limit_price or 100  # Use limit price or estimate
        order_value = order_request.quantity * estimated_price
        
        # Check maximum order value
        if order_value > self.max_order_value:
            errors.append(f"Order value ${order_value:,.2f} exceeds maximum ${self.max_order_value:,.2f}")
        
        # Check position size limits
        try:
            current_positions = await self.client.get_positions()
            current_position = 0
            
            for position in current_positions:
                if position.symbol == order_request.symbol:
                    current_position = position.position
                    break
            
            new_position = current_position
            if order_request.action == OrderAction.BUY:
                new_position += order_request.quantity
            else:
                new_position -= order_request.quantity
            
            if abs(new_position) > self.max_position_size:
                errors.append(f"New position size {new_position} exceeds maximum {self.max_position_size}")
                
        except Exception as e:
            logger.warning(f"Could not check position limits: {e}")
        
        return errors
    
    def _create_contract(self, order_request: OrderRequest) -> Contract:
        """Create IBKR contract from order request"""
        contract = Contract()
        contract.symbol = order_request.symbol
        contract.secType = order_request.security_type
        contract.exchange = order_request.exchange
        contract.currency = order_request.currency
        
        return contract
    
    def _create_ib_order(self, order_request: OrderRequest) -> Order:
        """Create IBKR order from order request"""
        order = Order()
        order.action = order_request.action.value
        order.totalQuantity = order_request.quantity
        order.orderType = order_request.order_type.value
        order.tif = order_request.time_in_force.value
        
        if order_request.limit_price:
            order.lmtPrice = order_request.limit_price
        
        if order_request.stop_price:
            order.auxPrice = order_request.stop_price
        
        if order_request.account:
            order.account = order_request.account
        
        return order
    
    async def order_target_percent(self, symbol: str, target_percent: float,
                                  order_type: OrderType = OrderType.MARKET,
                                  limit_price: Optional[float] = None,
                                  stop_price: Optional[float] = None,
                                  time_in_force: TimeInForce = TimeInForce.DAY,
                                  dry_run: bool = False) -> Dict[str, Any]:
        """
        Place an order to adjust a position to a target percent of the current portfolio value.
        
        This will automatically calculate whether to buy or sell to reach the target allocation.
        If the position doesn't exist, this creates a new position.
        If it exists, this adjusts the current position to the target percentage.
        
        Args:
            symbol: Stock symbol
            target_percent: Target allocation as decimal (0.1 = 10% of portfolio)
            order_type: Order type (MARKET, LIMIT, etc.)
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders  
            time_in_force: Time in force
            dry_run: If True, validate but don't place the order
            
        Returns:
            Order result dictionary
        """
        try:
            # Get current portfolio value
            positions = await self.client.get_positions()
            account_values = await self.client.get_account_values()
            
            # Find net liquidation value
            net_liquidation = 0
            for value in account_values:
                if value.key == 'NetLiquidation' and value.currency == 'USD':
                    net_liquidation = float(value.value)
                    break
            
            if net_liquidation <= 0:
                raise ValidationError("Could not determine portfolio value")
            
            # Calculate target value in dollars
            target_value = net_liquidation * target_percent
            
            # Get current market price for the symbol
            contract = self._create_contract(OrderRequest(
                symbol=symbol, action=OrderAction.BUY, quantity=1,
                order_type=order_type
            ))
            
            # Get current price via market data
            market_data = await self.client.get_market_data(contract, timeout=10.0)
            if not market_data or not market_data.get('last'):
                raise ValidationError(f"Could not get market price for {symbol}")
            
            current_price = float(market_data['last'])
            
            # Find current position
            current_position = 0
            current_value = 0
            
            for position in positions:
                if position.symbol == symbol:
                    current_position = position.position
                    current_value = current_position * current_price
                    break
            
            # Calculate required change
            value_difference = target_value - current_value
            quantity_needed = int(value_difference / current_price)
            
            # Determine action
            if quantity_needed == 0:
                return {
                    "status": "no_change_needed",
                    "message": f"Position already at target ({target_percent:.1%})",
                    "symbol": symbol,
                    "current_position": current_position,
                    "target_percent": target_percent,
                    "current_percent": current_value / net_liquidation
                }
            
            action = OrderAction.BUY if quantity_needed > 0 else OrderAction.SELL
            quantity = abs(quantity_needed)
            
            # Create order request
            order_request = OrderRequest(
                symbol=symbol,
                action=action,
                quantity=quantity,
                order_type=order_type,
                limit_price=limit_price,
                stop_price=stop_price,
                time_in_force=time_in_force
            )
            
            # Place the order
            result = await self.place_order(order_request, dry_run=dry_run)
            
            # Add portfolio context to result
            result['target_percent'] = target_percent
            result['current_position'] = current_position
            result['target_value'] = target_value
            result['current_value'] = current_value
            result['portfolio_value'] = net_liquidation
            
            logger.info(f"Target percent order for {symbol}: {action.value} {quantity} shares "
                       f"to reach {target_percent:.1%} allocation")
            
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
                           order_type: OrderType = OrderType.MARKET,
                           limit_price: Optional[float] = None,
                           stop_price: Optional[float] = None,
                           time_in_force: TimeInForce = TimeInForce.DAY,
                           dry_run: bool = False) -> Dict[str, Any]:
        """
        Place an order for a percentage of current portfolio value.
        
        This places a new order without considering existing positions.
        Use order_target_percent to adjust to a target allocation.
        
        Args:
            symbol: Stock symbol
            percent: Percentage of portfolio as decimal (0.1 = 10% of portfolio)
            order_type: Order type
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            time_in_force: Time in force
            dry_run: If True, validate but don't place the order
            
        Returns:
            Order result dictionary
        """
        try:
            # Get portfolio value
            account_values = await self.client.get_account_values()
            
            net_liquidation = 0
            for value in account_values:
                if value.key == 'NetLiquidation' and value.currency == 'USD':
                    net_liquidation = float(value.value)
                    break
            
            if net_liquidation <= 0:
                raise ValidationError("Could not determine portfolio value")
            
            # Calculate order value
            order_value = net_liquidation * abs(percent)
            
            # Get current market price
            contract = self._create_contract(OrderRequest(
                symbol=symbol, action=OrderAction.BUY, quantity=1,
                order_type=order_type
            ))
            
            market_data = await self.client.get_market_data(contract, timeout=10.0)
            if not market_data or not market_data.get('last'):
                raise ValidationError(f"Could not get market price for {symbol}")
            
            current_price = float(market_data['last'])
            quantity = int(order_value / current_price)
            
            if quantity <= 0:
                raise ValidationError(f"Calculated quantity too small: {quantity}")
            
            # Determine action based on percent sign
            action = OrderAction.BUY if percent > 0 else OrderAction.SELL
            
            # Create order request
            order_request = OrderRequest(
                symbol=symbol,
                action=action,
                quantity=quantity,
                order_type=order_type,
                limit_price=limit_price,
                stop_price=stop_price,
                time_in_force=time_in_force
            )
            
            result = await self.place_order(order_request, dry_run=dry_run)
            
            # Add context
            result['percent'] = percent
            result['order_value'] = order_value
            result['portfolio_value'] = net_liquidation
            result['price_used'] = current_price
            
            logger.info(f"Percent order for {symbol}: {action.value} {quantity} shares "
                       f"({percent:.1%} of portfolio)")
            
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
                                 order_type: OrderType = OrderType.MARKET,
                                 dry_run: bool = False) -> Dict[str, Any]:
        """
        Rebalance entire portfolio to target allocations.
        
        Args:
            target_allocations: Dict of {symbol: target_percent} where percents sum to <= 1.0
            order_type: Order type to use for all orders
            dry_run: If True, validate but don't place orders
            
        Returns:
            Dict with results for each symbol and summary
        """
        try:
            # Validate allocations
            total_allocation = sum(target_allocations.values())
            if total_allocation > 1.0:
                raise ValidationError(f"Total allocation {total_allocation:.1%} exceeds 100%")
            
            results = {}
            orders_placed = 0
            orders_failed = 0
            
            logger.info(f"Starting portfolio rebalance with {len(target_allocations)} targets")
            
            # Place orders for each target allocation
            for symbol, target_percent in target_allocations.items():
                try:
                    result = await self.order_target_percent(
                        symbol=symbol,
                        target_percent=target_percent,
                        order_type=order_type,
                        dry_run=dry_run
                    )
                    
                    results[symbol] = result
                    
                    if result['status'] in ['placed', 'validated']:
                        orders_placed += 1
                    elif result['status'] != 'no_change_needed':
                        orders_failed += 1
                        
                except Exception as e:
                    logger.error(f"Failed to rebalance {symbol}: {e}")
                    results[symbol] = {
                        "status": "failed",
                        "error": str(e)
                    }
                    orders_failed += 1
            
            # Summary
            summary = {
                "status": "completed",
                "total_symbols": len(target_allocations),
                "orders_placed": orders_placed,
                "orders_failed": orders_failed,
                "no_change_needed": len([r for r in results.values() 
                                         if r.get('status') == 'no_change_needed']),
                "total_target_allocation": total_allocation,
                "dry_run": dry_run
            }
            
            logger.info(f"Portfolio rebalance completed: {orders_placed} orders placed, "
                       f"{orders_failed} failed")
            
            return {
                "summary": summary,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Portfolio rebalance failed: {e}")
            return {
                "summary": {
                    "status": "failed",
                    "error": str(e),
                    "dry_run": dry_run
                },
                "results": {}
            }
    
    async def order_target_quantity(self, symbol: str, target_quantity: int,
                                   order_type: OrderType = OrderType.MARKET,
                                   limit_price: Optional[float] = None,
                                   stop_price: Optional[float] = None,
                                   time_in_force: TimeInForce = TimeInForce.DAY,
                                   dry_run: bool = False) -> Dict[str, Any]:
        """
        Place an order to adjust a position to a target number of shares.
        
        If the position doesn't exist, this creates a new position.
        If it exists, this adjusts to the exact target quantity.
        
        Args:
            symbol: Stock symbol
            target_quantity: Target number of shares (can be negative for short)
            order_type: Order type
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            time_in_force: Time in force
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
            
            # Calculate quantity needed
            quantity_needed = target_quantity - current_position
            
            if quantity_needed == 0:
                return {
                    "status": "no_change_needed",
                    "message": f"Position already at target ({target_quantity} shares)",
                    "symbol": symbol,
                    "current_position": current_position,
                    "target_quantity": target_quantity
                }
            
            # Determine action and absolute quantity
            action = OrderAction.BUY if quantity_needed > 0 else OrderAction.SELL
            quantity = abs(quantity_needed)
            
            # Create order request
            order_request = OrderRequest(
                symbol=symbol,
                action=action,
                quantity=quantity,
                order_type=order_type,
                limit_price=limit_price,
                stop_price=stop_price,
                time_in_force=time_in_force
            )
            
            # Place the order
            result = await self.place_order(order_request, dry_run=dry_run)
            
            # Add context
            result['target_quantity'] = target_quantity
            result['current_position'] = current_position
            result['quantity_needed'] = quantity_needed
            
            logger.info(f"Target quantity order for {symbol}: {action.value} {quantity} shares "
                       f"to reach {target_quantity} total shares")
            
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
                                order_type: OrderType = OrderType.MARKET,
                                limit_price: Optional[float] = None,
                                stop_price: Optional[float] = None,
                                time_in_force: TimeInForce = TimeInForce.DAY,
                                dry_run: bool = False) -> Dict[str, Any]:
        """
        Place an order to adjust a position to a target dollar value.
        
        Args:
            symbol: Stock symbol
            target_value: Target position value in dollars
            order_type: Order type
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            time_in_force: Time in force
            dry_run: If True, validate but don't place the order
            
        Returns:
            Order result dictionary
        """
        try:
            # Get current market price
            contract = self._create_contract(OrderRequest(
                symbol=symbol, action=OrderAction.BUY, quantity=1,
                order_type=order_type
            ))
            
            market_data = await self.client.get_market_data(contract, timeout=10.0)
            if not market_data or not market_data.get('last'):
                raise ValidationError(f"Could not get market price for {symbol}")
            
            current_price = float(market_data['last'])
            
            # Calculate target quantity
            target_quantity = int(target_value / current_price)
            
            # Use order_target_quantity to place the order
            result = await self.order_target_quantity(
                symbol=symbol,
                target_quantity=target_quantity,
                order_type=order_type,
                limit_price=limit_price,
                stop_price=stop_price,
                time_in_force=time_in_force,
                dry_run=dry_run
            )
            
            # Add value context
            result['target_value'] = target_value
            result['price_used'] = current_price
            
            logger.info(f"Target value order for {symbol}: targeting ${target_value:,.2f} "
                       f"(~{target_quantity} shares @ ${current_price:.2f})")
            
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
                         order_type: OrderType = OrderType.MARKET,
                         limit_price: Optional[float] = None,
                         stop_price: Optional[float] = None,
                         time_in_force: TimeInForce = TimeInForce.DAY,
                         dry_run: bool = False) -> Dict[str, Any]:
        """
        Place an order for a fixed dollar amount.
        
        Args:
            symbol: Stock symbol
            value: Dollar amount to trade (positive for buy, negative for sell)
            order_type: Order type
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            time_in_force: Time in force
            dry_run: If True, validate but don't place the order
            
        Returns:
            Order result dictionary
        """
        try:
            # Get current market price
            contract = self._create_contract(OrderRequest(
                symbol=symbol, action=OrderAction.BUY, quantity=1,
                order_type=order_type
            ))
            
            market_data = await self.client.get_market_data(contract, timeout=10.0)
            if not market_data or not market_data.get('last'):
                raise ValidationError(f"Could not get market price for {symbol}")
            
            current_price = float(market_data['last'])
            
            # Calculate quantity
            quantity = int(abs(value) / current_price)
            
            if quantity <= 0:
                raise ValidationError(f"Calculated quantity too small: {quantity}")
            
            # Determine action based on value sign
            action = OrderAction.BUY if value > 0 else OrderAction.SELL
            
            # Create order request
            order_request = OrderRequest(
                symbol=symbol,
                action=action,
                quantity=quantity,
                order_type=order_type,
                limit_price=limit_price,
                stop_price=stop_price,
                time_in_force=time_in_force
            )
            
            result = await self.place_order(order_request, dry_run=dry_run)
            
            # Add context
            result['order_value'] = value
            result['price_used'] = current_price
            result['calculated_quantity'] = quantity
            
            logger.info(f"Value order for {symbol}: {action.value} {quantity} shares "
                       f"for ${abs(value):,.2f}")
            
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
        try:
            # Use IBKR's global cancel function
            self.client.reqGlobalCancel()
            
            # Track what we cancelled
            cancelled_orders = list(self.active_orders.keys())
            
            # Move all active orders to history as cancelled
            for order_id in cancelled_orders:
                if order_id in self.active_orders:
                    tracker = self.active_orders[order_id]
                    tracker.status = "Cancelled"
                    tracker.updated_at = datetime.now()
                    self.order_history.append(self.active_orders.pop(order_id))
            
            logger.info(f"Cancelled all orders: {len(cancelled_orders)} orders")
            
            return {
                "status": "success",
                "cancelled_orders": cancelled_orders,
                "total_cancelled": len(cancelled_orders),
                "message": f"Cancelled {len(cancelled_orders)} orders"
            }
            
        except Exception as e:
            logger.error(f"Failed to cancel all orders: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def update_order(self, order_id: int, order_request: OrderRequest,
                          dry_run: bool = False) -> Dict[str, Any]:
        """
        Update an existing order by cancelling and replacing it.
        
        Args:
            order_id: Existing order ID to update
            order_request: New order parameters
            dry_run: If True, validate but don't actually update
            
        Returns:
            Update result dictionary
        """
        try:
            if order_id not in self.active_orders:
                return {
                    "status": "error",
                    "message": f"Order {order_id} not found in active orders"
                }
            
            if dry_run:
                # Validate new order
                validation_errors = order_request.validate()
                if validation_errors:
                    return {
                        "status": "validation_failed",
                        "errors": validation_errors
                    }
                
                return {
                    "status": "validated",
                    "message": "Order update validation successful",
                    "original_order_id": order_id,
                    "new_order": asdict(order_request)
                }
            
            # Cancel the original order
            cancel_result = await self.cancel_order(order_id)
            
            if cancel_result["status"] != "cancelled":
                return {
                    "status": "cancel_failed",
                    "message": f"Failed to cancel original order: {cancel_result.get('error')}",
                    "original_order_id": order_id
                }
            
            # Place the new order
            new_result = await self.place_order(order_request, dry_run=False)
            
            if new_result["status"] == "placed":
                logger.info(f"Order updated: cancelled {order_id}, placed {new_result.get('order_id')}")
                return {
                    "status": "updated",
                    "original_order_id": order_id,
                    "new_order_id": new_result.get("order_id"),
                    "message": "Order updated successfully"
                }
            else:
                return {
                    "status": "replace_failed",
                    "message": f"Cancelled original order but failed to place new order: {new_result.get('error')}",
                    "original_order_id": order_id
                }
            
        except Exception as e:
            logger.error(f"Failed to update order {order_id}: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "original_order_id": order_id
            }
    
    async def send_order(self, contract, order) -> int:
        """
        Send an order directly (low-level function).
        
        This is a lower-level function that bypasses validation and tracking.
        Use place_order() for full order management features.
        
        Args:
            contract: IBKR contract object
            order: IBKR order object
            
        Returns:
            Order ID
        """
        try:
            result = await self.client.place_order(contract, order)
            return result.get("order_id")
        except Exception as e:
            logger.error(f"Failed to send order: {e}")
            raise
    
    async def update_order_status(self, order_id: int, status_update: Dict[str, Any]):
        """Update order status from IBKR callbacks"""
        if order_id in self.active_orders:
            tracker = self.active_orders[order_id]
            tracker.status = status_update.get("status", tracker.status)
            tracker.filled = status_update.get("filled", tracker.filled)
            tracker.remaining = status_update.get("remaining", tracker.remaining)
            tracker.avg_fill_price = status_update.get("avg_fill_price", tracker.avg_fill_price)
            tracker.last_fill_price = status_update.get("last_fill_price", tracker.last_fill_price)
            tracker.updated_at = datetime.now()
            
            # If order is complete, move to history
            if tracker.status in ["Filled", "Cancelled"]:
                self.order_history.append(self.active_orders.pop(order_id))
                logger.info(f"Order {order_id} completed with status: {tracker.status}")
