#!/usr/bin/env python3
"""
Simple trading strategy example using the integrated IBKR application.

This demonstrates a basic momentum trading strategy that:
1. Monitors a watchlist of stocks
2. Calculates simple moving averages
3. Places buy/sell orders based on price crossovers
4. Manages position sizing and risk

This is for educational purposes only - not financial advice!
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List
import pandas as pd

from src.app.main import IBKRApp

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleMovingAverageStrategy:
    """Simple moving average crossover strategy"""
    
    def __init__(self, app: IBKRApp):
        self.app = app
        self.watchlist = ['AAPL', 'MSFT', 'GOOGL']  # Stocks to monitor
        self.short_window = 5   # Short MA period (5 minutes)
        self.long_window = 20   # Long MA period (20 minutes)
        self.position_size = 100  # Number of shares per trade
        self.max_positions = 3    # Maximum number of concurrent positions
        
        # Strategy state
        self.price_history: Dict[str, List[float]] = {}
        self.positions: Dict[str, int] = {}  # Current positions per symbol
        self.last_signals: Dict[str, str] = {}  # Last signal per symbol
        
        # Risk management
        self.max_portfolio_risk = 0.02  # 2% max risk per trade
        self.stop_loss_pct = 0.05       # 5% stop loss
        
    async def run(self, duration_minutes: int = 60):
        """Run the strategy for specified duration"""
        logger.info(f"Starting Simple MA Strategy for {duration_minutes} minutes...")
        logger.info(f"Watchlist: {self.watchlist}")
        logger.info(f"MA periods: {self.short_window}/{self.long_window}")
        
        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        
        # Initialize price history
        for symbol in self.watchlist:
            self.price_history[symbol] = []
            self.positions[symbol] = 0
            self.last_signals[symbol] = 'NONE'
        
        try:
            while datetime.now() < end_time:
                await self._strategy_loop()
                await asyncio.sleep(60)  # Run every minute
                
        except KeyboardInterrupt:
            logger.info("Strategy interrupted by user")
        except Exception as e:
            logger.error(f"Strategy error: {e}")
        finally:
            # Close all positions at end
            await self._close_all_positions()
    
    async def _strategy_loop(self):
        """Single iteration of the strategy"""
        try:
            # Update current positions from broker
            await self._update_positions()
            
            # Process each symbol in watchlist
            for symbol in self.watchlist:
                await self._process_symbol(symbol)
            
            # Show strategy status
            self._log_status()
            
        except Exception as e:
            logger.error(f"Error in strategy loop: {e}")
    
    async def _process_symbol(self, symbol: str):
        """Process trading signals for a single symbol"""
        try:
            # Get current market quote
            quote = await self.app.get_market_quote(symbol)
            if not quote or not quote.get('last'):
                logger.warning(f"No quote data for {symbol}")
                return
            
            current_price = float(quote['last'])\n            
            # Add to price history
            self.price_history[symbol].append(current_price)
            
            # Keep only the data we need (long_window + buffer)
            if len(self.price_history[symbol]) > self.long_window + 10:
                self.price_history[symbol] = self.price_history[symbol][-self.long_window - 5:]
            
            # Need enough data for long MA
            if len(self.price_history[symbol]) < self.long_window:
                logger.debug(f"{symbol}: Not enough data ({len(self.price_history[symbol])}/{self.long_window})")
                return
            
            # Calculate moving averages
            prices = pd.Series(self.price_history[symbol])
            short_ma = prices.rolling(window=self.short_window).mean().iloc[-1]
            long_ma = prices.rolling(window=self.long_window).mean().iloc[-1]
            
            # Determine signal
            signal = self._get_signal(short_ma, long_ma, symbol)
            
            # Act on signal
            if signal != 'NONE':
                await self._execute_signal(symbol, signal, current_price)
                
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
    
    def _get_signal(self, short_ma: float, long_ma: float, symbol: str) -> str:
        """Determine trading signal based on MA crossover"""
        if pd.isna(short_ma) or pd.isna(long_ma):
            return 'NONE'
        
        current_signal = 'BUY' if short_ma > long_ma else 'SELL'
        
        # Only signal on crossover (change from previous signal)
        if current_signal != self.last_signals[symbol]:
            logger.info(f"{symbol}: MA crossover detected - {current_signal} signal "
                       f"(Short MA: {short_ma:.2f}, Long MA: {long_ma:.2f})")
            self.last_signals[symbol] = current_signal
            return current_signal
        
        return 'NONE'
    
    async def _execute_signal(self, symbol: str, signal: str, current_price: float):
        """Execute trading signal"""
        try:
            current_position = self.positions[symbol]
            
            if signal == 'BUY' and current_position <= 0:
                # Buy signal and no long position
                if self._can_add_position():
                    quantity = self._calculate_position_size(current_price)
                    if quantity > 0:
                        result = await self.app.place_buy_order(
                            symbol=symbol,
                            quantity=quantity,
                            order_type="MARKET",
                            dry_run=False  # Set to True for paper trading
                        )
                        
                        if result['status'] == 'placed':
                            logger.info(f"✅ BUY order placed: {symbol} x{quantity} @ Market")
                            # Update our position tracking (will be confirmed later)
                            self.positions[symbol] += quantity
                        else:
                            logger.error(f"❌ BUY order failed for {symbol}: {result.get('error')}")
            
            elif signal == 'SELL' and current_position > 0:
                # Sell signal and have long position
                result = await self.app.place_sell_order(
                    symbol=symbol,
                    quantity=current_position,  # Sell entire position
                    order_type="MARKET",
                    dry_run=False  # Set to True for paper trading
                )
                
                if result['status'] == 'placed':
                    logger.info(f"✅ SELL order placed: {symbol} x{current_position} @ Market")
                    # Update our position tracking
                    self.positions[symbol] = 0
                else:
                    logger.error(f"❌ SELL order failed for {symbol}: {result.get('error')}")
            
        except Exception as e:
            logger.error(f"Error executing {signal} signal for {symbol}: {e}")
    
    def _can_add_position(self) -> bool:
        """Check if we can add another position (risk management)"""
        active_positions = sum(1 for pos in self.positions.values() if pos > 0)
        return active_positions < self.max_positions
    
    def _calculate_position_size(self, price: float) -> int:
        """Calculate position size based on risk management"""
        try:
            # Simple position sizing - could be more sophisticated
            # For demo, use fixed position size, but respect available capital
            
            # Get current account info
            portfolio = asyncio.create_task(self.app.get_portfolio_summary())
            # This is simplified - in real implementation, you'd want to cache this
            
            return self.position_size  # Fixed size for simplicity
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0
    
    async def _update_positions(self):
        """Update position tracking from actual broker positions"""
        try:
            # Get actual positions from broker
            portfolio = await self.app.get_portfolio_summary()
            broker_positions = portfolio.get('positions', {}).get('positions', [])
            
            # Update our tracking
            for symbol in self.watchlist:
                self.positions[symbol] = 0  # Reset
                
            for pos in broker_positions:
                symbol = pos['symbol']
                if symbol in self.watchlist:
                    self.positions[symbol] = int(pos['quantity'])
                    
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
    async def _close_all_positions(self):
        """Close all open positions"""
        logger.info("Closing all positions...")
        
        for symbol in self.watchlist:
            if self.positions[symbol] > 0:
                try:
                    result = await self.app.close_position(symbol, dry_run=False)
                    if result['status'] in ['placed', 'no_position']:
                        logger.info(f"✅ Closed position in {symbol}")
                    else:
                        logger.error(f"❌ Failed to close {symbol}: {result.get('error')}")
                except Exception as e:
                    logger.error(f"Error closing {symbol}: {e}")
    
    def _log_status(self):
        """Log current strategy status"""
        active_positions = sum(1 for pos in self.positions.values() if pos > 0)
        total_shares = sum(pos for pos in self.positions.values() if pos > 0)
        
        if active_positions > 0:
            logger.info(f"Strategy Status: {active_positions} positions, {total_shares} total shares")
            for symbol, position in self.positions.items():
                if position > 0:
                    logger.info(f"  {symbol}: {position} shares")


async def main():
    """Main function to run the strategy"""
    
    # Initialize the trading application
    app = IBKRApp()
    
    try:
        logger.info("Starting IBKR Trading Application for Strategy...")
        
        # Start the application
        if not await app.start():
            logger.error("Failed to start trading application")
            return
        
        logger.info("✅ Application started successfully!")
        
        # Wait for connection to stabilize
        await asyncio.sleep(3)
        
        # Create and run the strategy
        strategy = SimpleMovingAverageStrategy(app)
        
        # Run strategy for 30 minutes (change as needed)
        await strategy.run(duration_minutes=30)
        
        logger.info("Strategy completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Strategy interrupted by user")
    except Exception as e:
        logger.error(f"Strategy failed: {e}")
    finally:
        # Clean shutdown
        logger.info("Shutting down application...")
        await app.stop()
        logger.info("✅ Application shutdown complete")


if __name__ == "__main__":
    print("Simple Moving Average Trading Strategy")
    print("=" * 40)
    print()
    print("This strategy demonstrates:")
    print("• Real-time price monitoring")
    print("• Moving average calculations")
    print("• Automated order placement")
    print("• Position management")
    print("• Risk controls")
    print()
    print("⚠️  WARNING: This will place REAL orders!")
    print("   Set dry_run=True in the code for paper trading")
    print()
    
    confirm = input("Are you sure you want to run live trading? (yes/no): ")
    if confirm.lower() != 'yes':
        print("Aborted - good choice for safety!")
        exit(0)
    
    print()
    print("Starting strategy...")
    
    asyncio.run(main())
