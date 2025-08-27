"""
Async IBKR API Client with event-driven architecture.
Combines EClient and EWrapper functionality with modern async patterns.
"""

import asyncio
import logging
from typing import Dict, Optional, List, AsyncGenerator, Any, Callable
from collections import defaultdict
from datetime import datetime
import threading
from queue import Queue
from dataclasses import dataclass

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.common import TickerId, OrderId

from ..core.config import IBConfig
from ..core.types import MarketDataType, OrderStatus, PositionData, AccountValue
from ..core.exceptions import TimeoutError, ValidationError
from .connection import ConnectionManager

logger = logging.getLogger(__name__)


@dataclass
class PendingRequest:
    """Tracks pending API requests"""
    request_id: int
    request_type: str
    future: asyncio.Future
    timeout_handle: Optional[asyncio.Handle] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class IBWrapper(EWrapper):
    """Enhanced EWrapper with async event handling"""
    
    def __init__(self, client_instance):
        super().__init__()
        self.client = client_instance
        
        # Event queues for different data types
        self.market_data_queue = asyncio.Queue()
        self.order_status_queue = asyncio.Queue()
        self.position_queue = asyncio.Queue()
        self.account_queue = asyncio.Queue()
        self.error_queue = asyncio.Queue()
        
        # Active streaming subscriptions
        self.streaming_subscriptions: Dict[int, str] = {}
    
    def connectAck(self):
        """Connection acknowledgment"""
        logger.info("Connection acknowledged by IBKR")
        self.client.connection_manager.on_connected()
    
    def connectionClosed(self):
        """Connection closed"""
        logger.info("Connection closed by IBKR")
        self.client.connection_manager.on_disconnected()
    
    def error(self, reqId: TickerId, errorCode: int, errorString: str, advancedOrderRejectJson: str = ""):
        """Error handling"""
        error_info = {
            "request_id": reqId,
            "error_code": errorCode,
            "error_string": errorString,
            "advanced_order_reject_json": advancedOrderRejectJson,
            "timestamp": datetime.now()
        }
        
        logger.error(f"IBKR Error {errorCode}: {errorString} (ReqId: {reqId})")
        
        # Put error in queue for async handling
        try:
            self.error_queue.put_nowait(error_info)
        except asyncio.QueueFull:
            logger.warning("Error queue is full, dropping error message")
        
        # Handle specific request errors
        if reqId != -1 and reqId in self.client.pending_requests:
            request = self.client.pending_requests[reqId]
            if not request.future.done():
                request.future.set_exception(Exception(f"IBKR Error {errorCode}: {errorString}"))
    
    def tickPrice(self, reqId: TickerId, tickType: int, price: float, attrib):
        """Market data tick price"""
        tick_data = {
            "request_id": reqId,
            "tick_type": tickType,
            "price": price,
            "attrib": attrib,
            "timestamp": datetime.now(),
            "data_type": "price"
        }
        
        try:
            self.market_data_queue.put_nowait(tick_data)
        except asyncio.QueueFull:
            logger.warning("Market data queue is full, dropping tick data")
    
    def tickSize(self, reqId: TickerId, tickType: int, size: int):
        """Market data tick size"""
        tick_data = {
            "request_id": reqId,
            "tick_type": tickType,
            "size": size,
            "timestamp": datetime.now(),
            "data_type": "size"
        }
        
        try:
            self.market_data_queue.put_nowait(tick_data)
        except asyncio.QueueFull:
            logger.warning("Market data queue is full, dropping tick data")
    
    def orderStatus(self, orderId: OrderId, status: str, filled: float,
                   remaining: float, avgFillPrice: float, permId: int,
                   parentId: int, lastFillPrice: float, clientId: int,
                   whyHeld: str, mktCapPrice: float):
        """Order status update"""
        order_status = {
            "order_id": orderId,
            "status": status,
            "filled": filled,
            "remaining": remaining,
            "avg_fill_price": avgFillPrice,
            "perm_id": permId,
            "parent_id": parentId,
            "last_fill_price": lastFillPrice,
            "client_id": clientId,
            "why_held": whyHeld,
            "mkt_cap_price": mktCapPrice,
            "timestamp": datetime.now()
        }
        
        try:
            self.order_status_queue.put_nowait(order_status)
        except asyncio.QueueFull:
            logger.warning("Order status queue is full, dropping order status")
    
    def position(self, account: str, contract: Contract, position: float, avgCost: float):
        """Position update"""
        position_data = PositionData(
            account=account,
            symbol=contract.symbol,
            sec_type=contract.secType,
            exchange=contract.exchange,
            currency=contract.currency,
            position=position,
            avg_cost=avgCost,
            timestamp=datetime.now()
        )
        
        try:
            self.position_queue.put_nowait(position_data)
        except asyncio.QueueFull:
            logger.warning("Position queue is full, dropping position data")
    
    def accountValue(self, key: str, val: str, currency: str, accountName: str):
        """Account value update"""
        account_value = AccountValue(
            key=key,
            value=val,
            currency=currency,
            account=accountName,
            timestamp=datetime.now()
        )
        
        try:
            self.account_queue.put_nowait(account_value)
        except asyncio.QueueFull:
            logger.warning("Account queue is full, dropping account value")


class IBClient(EClient):
    """Async IBKR API Client"""
    
    def __init__(self, config: IBConfig):
        if config is None:
            raise ValueError("Config cannot be None")
        
        self.wrapper = IBWrapper(self)
        super().__init__(self.wrapper)
        
        self.config = config
        self.connection_manager = ConnectionManager(config)
        self.connection_manager.set_client(self)
        
        # Request management
        self.next_request_id = 1
        self.pending_requests: Dict[int, PendingRequest] = {}
        self._request_lock = asyncio.Lock()
        
        # Background tasks
        self._message_processing_task: Optional[asyncio.Task] = None
        self._error_handling_task: Optional[asyncio.Task] = None
        
        # Event loop and thread management
        self._loop = None
        self._thread = None
        self._running = False
    
    async def start(self) -> bool:
        """Start the client and establish connection"""
        if self._running:
            logger.warning("Client is already running")
            return True
        
        self._loop = asyncio.get_event_loop()
        self._running = True
        
        # Start the client in a separate thread for socket processing
        self._thread = threading.Thread(target=self._run_socket_thread, daemon=True)
        self._thread.start()
        
        # Start background tasks
        self._message_processing_task = asyncio.create_task(self._process_messages())
        self._error_handling_task = asyncio.create_task(self._handle_errors())
        
        # Attempt connection
        success = await self.connection_manager.connect_with_retry()
        
        if success:
            logger.info("IBKR client started successfully")
        else:
            logger.error("Failed to start IBKR client")
            await self.stop()
        
        return success
    
    def _run_socket_thread(self):
        """Run the socket processing in a separate thread"""
        while self._running:
            try:
                self.run()
            except Exception as e:
                logger.error(f"Error in socket thread: {e}")
                if self._running:
                    # Schedule reconnection
                    if self._loop and not self._loop.is_closed():
                        asyncio.run_coroutine_threadsafe(
                            self.connection_manager._handle_connection_loss(),
                            self._loop
                        )
                break
    
    async def stop(self):
        """Stop the client and disconnect"""
        if not self._running:
            return
        
        logger.info("Stopping IBKR client...")
        self._running = False
        
        # Cancel all pending requests
        for request in self.pending_requests.values():
            if not request.future.done():
                request.future.cancel()
            if request.timeout_handle:
                request.timeout_handle.cancel()
        
        self.pending_requests.clear()
        
        # Stop background tasks
        if self._message_processing_task:
            self._message_processing_task.cancel()
            try:
                await self._message_processing_task
            except asyncio.CancelledError:
                pass
        
        if self._error_handling_task:
            self._error_handling_task.cancel()
            try:
                await self._error_handling_task
            except asyncio.CancelledError:
                pass
        
        # Disconnect
        await self.connection_manager.disconnect()
        
        # Wait for thread to finish
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        
        logger.info("IBKR client stopped")
    
    async def _get_next_request_id(self) -> int:
        """Get next available request ID"""
        async with self._request_lock:
            request_id = self.next_request_id
            self.next_request_id += 1
            return request_id
    
    async def _create_request(self, request_type: str, timeout: Optional[float] = None) -> PendingRequest:
        """Create a new pending request"""
        request_id = await self._get_next_request_id()
        future = asyncio.Future()
        
        request = PendingRequest(
            request_id=request_id,
            request_type=request_type,
            future=future
        )
        
        # Set timeout if specified
        if timeout:
            def timeout_callback():
                if not future.done():
                    future.set_exception(TimeoutError(f"Request {request_id} timed out after {timeout} seconds"))
            
            request.timeout_handle = asyncio.get_event_loop().call_later(timeout, timeout_callback)
        
        self.pending_requests[request_id] = request
        return request
    
    def _complete_request(self, request_id: int, result: Any = None):
        """Complete a pending request"""
        if request_id in self.pending_requests:
            request = self.pending_requests.pop(request_id)
            if not request.future.done():
                request.future.set_result(result)
            if request.timeout_handle:
                request.timeout_handle.cancel()
    
    async def _process_messages(self):
        """Background task to process various message queues"""
        while self._running:
            try:
                # Process with a small timeout to allow cancellation
                await asyncio.sleep(0.01)
                
                # Additional message processing logic can be added here
                # For now, most processing is handled in the wrapper callbacks
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in message processing: {e}")
                await asyncio.sleep(1)
    
    async def _handle_errors(self):
        """Background task to handle errors from the error queue"""
        while self._running:
            try:
                # Wait for error with timeout to allow cancellation
                error_info = await asyncio.wait_for(
                    self.wrapper.error_queue.get(),
                    timeout=1.0
                )
                
                # Handle specific error codes here if needed
                error_code = error_info.get("error_code")
                request_id = error_info.get("request_id")
                
                # Log error details
                logger.error(f"Handling error {error_code} for request {request_id}")
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in error handling: {e}")
                await asyncio.sleep(1)
    
    async def get_market_data(self, contract: Contract, 
                            market_data_type: MarketDataType = MarketDataType.REAL_TIME,
                            timeout: float = 10.0) -> Dict[str, Any]:
        """Get snapshot market data"""
        if not self.connection_manager.is_connected:
            raise ConnectionError("Not connected to IBKR")
        
        request = await self._create_request("market_data_snapshot", timeout)
        
        # Request market data snapshot
        self.reqMktData(
            request.request_id,
            contract,
            "",  # generic tick list
            True,  # snapshot
            False,  # regulatory snapshot
            []  # mkt data options
        )
        
        try:
            result = await request.future
            return result
        finally:
            # Cancel market data request
            self.cancelMktData(request.request_id)
    
    async def get_streaming_data(self, contract: Contract, 
                               market_data_type: MarketDataType = MarketDataType.REAL_TIME) -> AsyncGenerator[Dict[str, Any], None]:
        """Get streaming market data"""
        if not self.connection_manager.is_connected:
            raise ConnectionError("Not connected to IBKR")
        
        request_id = await self._get_next_request_id()
        
        # Start streaming market data
        self.reqMktData(
            request_id,
            contract,
            "",  # generic tick list
            False,  # not a snapshot
            False,  # regulatory snapshot
            []  # mkt data options
        )
        
        # Track streaming subscription
        self.wrapper.streaming_subscriptions[request_id] = "market_data"
        
        try:
            while True:
                try:
                    # Wait for market data with timeout
                    tick_data = await asyncio.wait_for(
                        self.wrapper.market_data_queue.get(),
                        timeout=30.0  # 30 second timeout for streaming data
                    )
                    
                    # Only yield data for this request
                    if tick_data.get("request_id") == request_id:
                        yield tick_data
                    else:
                        # Put it back if it's for a different request
                        await self.wrapper.market_data_queue.put(tick_data)
                
                except asyncio.TimeoutError:
                    logger.warning(f"Market data timeout for request {request_id}")
                    break
                
        finally:
            # Cancel streaming subscription
            self.cancelMktData(request_id)
            if request_id in self.wrapper.streaming_subscriptions:
                del self.wrapper.streaming_subscriptions[request_id]
    
    async def place_order(self, contract: Contract, order: Order, timeout: float = 30.0) -> Dict[str, Any]:
        """Place an order"""
        if not self.connection_manager.is_connected:
            raise ConnectionError("Not connected to IBKR")
        
        order_id = await self._get_next_request_id()
        request = await self._create_request("place_order", timeout)
        
        # Place the order
        self.placeOrder(order_id, contract, order)
        
        try:
            result = await request.future
            return {"order_id": order_id, "status": result}
        except Exception:
            # Cancel the order if something went wrong
            self.cancelOrder(order_id)
            raise
    
    async def get_positions(self, timeout: float = 10.0) -> List[PositionData]:
        """Get current positions"""
        if not self.connection_manager.is_connected:
            raise ConnectionError("Not connected to IBKR")
        
        request = await self._create_request("positions", timeout)
        
        # Request positions
        self.reqPositions()
        
        try:
            positions = []
            start_time = datetime.now()
            
            while (datetime.now() - start_time).total_seconds() < timeout:
                try:
                    position = await asyncio.wait_for(
                        self.wrapper.position_queue.get(),
                        timeout=1.0
                    )
                    positions.append(position)
                except asyncio.TimeoutError:
                    # Check if we have any positions
                    if positions:
                        break
                    continue
            
            self._complete_request(request.request_id, positions)
            return positions
            
        finally:
            # Cancel positions request
            self.cancelPositions()
    
    async def get_account_values(self, account: str = "", timeout: float = 10.0) -> List[AccountValue]:
        """Get account values"""
        if not self.connection_manager.is_connected:
            raise ConnectionError("Not connected to IBKR")
        
        request = await self._create_request("account_values", timeout)
        
        # Request account updates
        self.reqAccountUpdates(True, account)
        
        try:
            values = []
            start_time = datetime.now()
            
            while (datetime.now() - start_time).total_seconds() < timeout:
                try:
                    account_value = await asyncio.wait_for(
                        self.wrapper.account_queue.get(),
                        timeout=1.0
                    )
                    values.append(account_value)
                except asyncio.TimeoutError:
                    # Check if we have any values
                    if values:
                        break
                    continue
            
            self._complete_request(request.request_id, values)
            return values
            
        finally:
            # Cancel account updates
            self.reqAccountUpdates(False, account)
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to IBKR"""
        return self.connection_manager.is_connected
    
    async def get_pnl(self, account: str = "", timeout: float = 10.0) -> Dict[str, Any]:
        """
        Get current P&L information
        
        Args:
            account: Account name (if empty, uses default account)
            timeout: Request timeout
            
        Returns:
            Dict with P&L data
        """
        if not self.connection_manager.is_connected:
            raise ConnectionError("Not connected to IBKR")
        
        request_id = await self._get_next_request_id()
        request = await self._create_request("pnl", timeout)
        
        # Request P&L
        self.reqPnL(request_id, account, "")
        
        try:
            # Wait for P&L data
            start_time = datetime.now()
            pnl_data = {}
            
            while (datetime.now() - start_time).total_seconds() < timeout:
                try:
                    # This would need to be implemented in the wrapper
                    # For now, return basic structure
                    await asyncio.sleep(0.1)
                    break
                except asyncio.TimeoutError:
                    continue
            
            return {
                "request_id": request_id,
                "account": account,
                "daily_pnl": 0.0,  # Would come from actual PnL callback
                "unrealized_pnl": 0.0,
                "realized_pnl": 0.0,
                "timestamp": datetime.now()
            }
            
        finally:
            # Cancel P&L subscription
            self.cancelPnL(request_id)
    
    def cancel_all_orders(self):
        """
        Cancel all orders for the account
        """
        if not self.connection_manager.is_connected:
            raise ConnectionError("Not connected to IBKR")
        
        self.reqGlobalCancel()
        logger.info("Requested cancellation of all orders")
    
    def cancel_order_by_id(self, order_id: int):
        """
        Cancel a specific order by ID
        
        Args:
            order_id: Order ID to cancel
        """
        if not self.connection_manager.is_connected:
            raise ConnectionError("Not connected to IBKR")
        
        self.cancelOrder(order_id)
        logger.info(f"Requested cancellation of order {order_id}")
    
    async def get_historical_data(self, contract: Contract, duration: str, 
                                 bar_size: str, what_to_show: str = "TRADES",
                                 timeout: float = 30.0) -> List[Dict[str, Any]]:
        """
        Get historical data for a contract
        
        Args:
            contract: Contract to get data for
            duration: Duration string (e.g., "1 D", "1 W", "1 M")
            bar_size: Bar size (e.g., "1 min", "5 mins", "1 day")
            what_to_show: What data to show (TRADES, MIDPOINT, BID, ASK)
            timeout: Request timeout
            
        Returns:
            List of historical bars
        """
        if not self.connection_manager.is_connected:
            raise ConnectionError("Not connected to IBKR")
        
        request_id = await self._get_next_request_id()
        request = await self._create_request("historical_data", timeout)
        
        # Request historical data
        self.reqHistoricalData(
            request_id,
            contract,
            "",  # end date (empty = now)
            duration,
            bar_size,
            what_to_show,
            1,  # useRTH
            1,  # formatDate
            False,  # keepUpToDate
            []  # chartOptions
        )
        
        try:
            # This would need proper implementation with historical data callbacks
            result = await request.future
            return result if isinstance(result, list) else []
        except Exception as e:
            logger.error(f"Failed to get historical data: {e}")
            return []
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to IBKR"""
        return self.connection_manager.is_connected
    
    @property
    def connection_info(self) -> dict:
        """Get connection information"""
        return self.connection_manager.connection_info
