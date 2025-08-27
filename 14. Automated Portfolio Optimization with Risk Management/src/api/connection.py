"""
Connection management for Interactive Brokers API.
Handles connection lifecycle with retry logic and health monitoring.
"""

import asyncio
import logging
from typing import Optional, Callable
from enum import Enum
import time
from datetime import datetime, timedelta

from ibapi.client import EClient

from ..core.config import IBConfig
from ..core.exceptions import ConnectionError, TimeoutError

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Connection state enumeration"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


class ConnectionManager:
    """Manages IBKR API connection with retry logic and health monitoring"""
    
    def __init__(self, config: IBConfig):
        self.config = config
        self.client: Optional[EClient] = None
        self.state = ConnectionState.DISCONNECTED
        self.connection_event = asyncio.Event()
        self.disconnect_event = asyncio.Event()
        
        # Connection tracking
        self.connect_time: Optional[datetime] = None
        self.last_heartbeat: Optional[datetime] = None
        self.connection_attempts = 0
        self.consecutive_failures = 0
        
        # Event callbacks
        self.on_connected_callback: Optional[Callable] = None
        self.on_disconnected_callback: Optional[Callable] = None
        self.on_error_callback: Optional[Callable] = None
        
        # Background tasks
        self._health_check_task: Optional[asyncio.Task] = None
        self._connection_monitor_task: Optional[asyncio.Task] = None
    
    def set_client(self, client: EClient):
        """Set the EClient instance to manage"""
        self.client = client
    
    def set_callbacks(self, 
                     on_connected: Optional[Callable] = None,
                     on_disconnected: Optional[Callable] = None,
                     on_error: Optional[Callable] = None):
        """Set connection event callbacks"""
        self.on_connected_callback = on_connected
        self.on_disconnected_callback = on_disconnected
        self.on_error_callback = on_error
    
    async def connect_with_retry(self) -> bool:
        """Connect with exponential backoff retry logic"""
        if not self.client:
            raise ConnectionError("No client set for connection manager")
        
        logger.info(f"Attempting to connect to IBKR at {self.config.host}:{self.config.port}")
        
        for attempt in range(1, self.config.max_reconnect_attempts + 1):
            self.connection_attempts = attempt
            self.state = ConnectionState.CONNECTING
            
            try:
                success = await self._single_connect_attempt()
                
                if success:
                    self.consecutive_failures = 0
                    self.state = ConnectionState.CONNECTED
                    self.connect_time = datetime.now()
                    
                    # Start health monitoring
                    await self._start_monitoring()
                    
                    logger.info(f"Successfully connected to IBKR on attempt {attempt}")
                    
                    if self.on_connected_callback:
                        try:
                            await self.on_connected_callback()
                        except Exception as e:
                            logger.error(f"Error in connection callback: {e}")
                    
                    return True
                
            except Exception as e:
                self.consecutive_failures += 1
                logger.warning(f"Connection attempt {attempt} failed: {e}")
                
                if self.on_error_callback:
                    try:
                        await self.on_error_callback(e)
                    except Exception as callback_error:
                        logger.error(f"Error in error callback: {callback_error}")
            
            # Calculate retry delay with exponential backoff
            if attempt < self.config.max_reconnect_attempts:
                delay = min(
                    self.config.reconnect_delay * (2 ** (attempt - 1)),
                    60.0  # Maximum 60 seconds delay
                )
                logger.info(f"Retrying connection in {delay:.1f} seconds...")
                await asyncio.sleep(delay)
        
        self.state = ConnectionState.FAILED
        logger.error(f"Failed to connect after {self.config.max_reconnect_attempts} attempts")
        return False
    
    async def _single_connect_attempt(self) -> bool:
        """Single connection attempt with timeout"""
        try:
            # Reset connection events
            self.connection_event.clear()
            
            # Attempt connection
            self.client.connect(
                self.config.host,
                self.config.port,
                self.config.client_id
            )
            
            # Wait for connection establishment with timeout
            connected = await asyncio.wait_for(
                self.connection_event.wait(),
                timeout=self.config.connection_timeout
            )
            
            return connected
            
        except asyncio.TimeoutError:
            logger.warning(f"Connection attempt timed out after {self.config.connection_timeout} seconds")
            if self.client.isConnected():
                self.client.disconnect()
            return False
        except Exception as e:
            logger.error(f"Connection attempt failed: {e}")
            if self.client.isConnected():
                self.client.disconnect()
            raise ConnectionError(f"Failed to connect: {e}")
    
    async def disconnect(self):
        """Gracefully disconnect from IBKR"""
        if self.state in [ConnectionState.DISCONNECTED, ConnectionState.FAILED]:
            return
        
        logger.info("Disconnecting from IBKR...")
        self.state = ConnectionState.DISCONNECTED
        
        # Stop monitoring tasks
        await self._stop_monitoring()
        
        # Disconnect client
        if self.client and self.client.isConnected():
            try:
                self.client.disconnect()
                
                # Wait for disconnection or timeout
                try:
                    await asyncio.wait_for(
                        self.disconnect_event.wait(),
                        timeout=5.0
                    )
                except asyncio.TimeoutError:
                    logger.warning("Disconnect acknowledgment timed out")
                
            except Exception as e:
                logger.error(f"Error during disconnect: {e}")
        
        # Call disconnection callback
        if self.on_disconnected_callback:
            try:
                await self.on_disconnected_callback()
            except Exception as e:
                logger.error(f"Error in disconnection callback: {e}")
        
        logger.info("Disconnected from IBKR")
    
    async def _start_monitoring(self):
        """Start background monitoring tasks"""
        # Start health check task
        if not self._health_check_task or self._health_check_task.done():
            self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        # Start connection monitor task
        if not self._connection_monitor_task or self._connection_monitor_task.done():
            self._connection_monitor_task = asyncio.create_task(self._connection_monitor_loop())
    
    async def _stop_monitoring(self):
        """Stop background monitoring tasks"""
        # Cancel health check task
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Cancel connection monitor task
        if self._connection_monitor_task and not self._connection_monitor_task.done():
            self._connection_monitor_task.cancel()
            try:
                await self._connection_monitor_task
            except asyncio.CancelledError:
                pass
    
    async def _health_check_loop(self):
        """Background task to monitor connection health"""
        while self.state == ConnectionState.CONNECTED:
            try:
                # Check if client is still connected
                if not self.client or not self.client.isConnected():
                    logger.warning("Connection lost - client reports disconnected")
                    await self._handle_connection_loss()
                    break
                
                # Update heartbeat
                self.last_heartbeat = datetime.now()
                
                # Check connection age and quality metrics could go here
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(10)
    
    async def _connection_monitor_loop(self):
        """Background task to monitor connection and handle reconnection"""
        while self.state in [ConnectionState.CONNECTED, ConnectionState.RECONNECTING]:
            try:
                # Monitor for unexpected disconnections
                if self.state == ConnectionState.CONNECTED:
                    # Check if we've lost connection
                    if (not self.client or 
                        not self.client.isConnected() or
                        (self.last_heartbeat and 
                         datetime.now() - self.last_heartbeat > timedelta(seconds=30))):
                        
                        logger.warning("Connection health check failed")
                        await self._handle_connection_loss()
                
                await asyncio.sleep(5)  # Monitor every 5 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in connection monitor loop: {e}")
                await asyncio.sleep(5)
    
    async def _handle_connection_loss(self):
        """Handle unexpected connection loss"""
        if self.state == ConnectionState.DISCONNECTED:
            return
        
        logger.warning("Connection lost - attempting to reconnect...")
        self.state = ConnectionState.RECONNECTING
        
        # Call disconnection callback
        if self.on_disconnected_callback:
            try:
                await self.on_disconnected_callback()
            except Exception as e:
                logger.error(f"Error in disconnection callback: {e}")
        
        # Attempt reconnection
        success = await self.connect_with_retry()
        
        if not success:
            self.state = ConnectionState.FAILED
            logger.error("Failed to reconnect to IBKR")
    
    def on_connected(self):
        """Called when connection is established (callback from wrapper)"""
        if self.state in [ConnectionState.CONNECTING, ConnectionState.RECONNECTING]:
            self.connection_event.set()
    
    def on_disconnected(self):
        """Called when connection is lost (callback from wrapper)"""
        self.disconnect_event.set()
        if self.state == ConnectionState.CONNECTED:
            # This was an unexpected disconnection
            asyncio.create_task(self._handle_connection_loss())
    
    @property
    def is_connected(self) -> bool:
        """Check if currently connected"""
        return (self.state == ConnectionState.CONNECTED and 
                self.client and 
                self.client.isConnected())
    
    @property
    def connection_info(self) -> dict:
        """Get connection information"""
        return {
            "state": self.state.value,
            "connected": self.is_connected,
            "host": self.config.host,
            "port": self.config.port,
            "client_id": self.config.client_id,
            "connect_time": self.connect_time.isoformat() if self.connect_time else None,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "connection_attempts": self.connection_attempts,
            "consecutive_failures": self.consecutive_failures,
            "uptime_seconds": (
                (datetime.now() - self.connect_time).total_seconds() 
                if self.connect_time and self.is_connected 
                else 0
            )
        }
    
    async def wait_for_connection(self, timeout: Optional[float] = None) -> bool:
        """Wait for connection to be established"""
        if self.is_connected:
            return True
        
        try:
            if timeout:
                await asyncio.wait_for(self.connection_event.wait(), timeout=timeout)
            else:
                await self.connection_event.wait()
            return True
        except asyncio.TimeoutError:
            return False
    
    async def ensure_connection(self) -> bool:
        """Ensure connection is active, reconnect if needed"""
        if self.is_connected:
            return True
        
        if self.state in [ConnectionState.DISCONNECTED, ConnectionState.FAILED]:
            return await self.connect_with_retry()
        
        # If connecting or reconnecting, wait for result
        return await self.wait_for_connection(timeout=self.config.connection_timeout)
