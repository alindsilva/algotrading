#!/usr/bin/env python3
"""
Portfolio Allocation Example

This script demonstrates how to use the order_target_percent function and portfolio
allocation features in the IBKR trading application.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any

from src.app.main import IBKRApp

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def show_portfolio_allocations(app: IBKRApp):
    """Show current portfolio allocations"""
    print("\n=== Current Portfolio Allocations ===")
    
    allocations = await app.get_portfolio_allocations()
    
    if not allocations:
        print("No positions found")
        return
    
    # Get total allocation
    total_allocation = sum(allocations.values())
    
    # Sort by allocation percentage (descending)
    sorted_allocations = sorted(allocations.items(), key=lambda x: x[1], reverse=True)
    
    print(f"{'Symbol':<8} {'Allocation':<12} {'Percent':<10}")
    print("-" * 30)
    
    for symbol, percent in sorted_allocations:
        print(f"{symbol:<8} {percent*100:>8.2f}%     {percent/total_allocation*100:>6.2f}%")
    
    print(f"\nTotal allocation: {total_allocation*100:.2f}%")
    print(f"Cash: {(1-total_allocation)*100:.2f}%")


async def example_target_percent_order(app: IBKRApp):
    """Demonstrate target percent order"""
    print("\n=== Order Target Percent Example ===")
    
    # First check current portfolio allocations
    await show_portfolio_allocations(app)
    
    # Example: Target 10% allocation in AAPL (adjust as needed)
    symbol = "AAPL"
    target_percent = 0.10  # 10%
    
    print(f"\nAttempting to set {symbol} to {target_percent*100:.1f}% allocation")
    
    # First do a dry run to check what would happen
    result = await app.order_target_percent(
        symbol=symbol, 
        target_percent=target_percent,
        dry_run=True  # Don't actually place the order yet
    )
    
    # Show the result of the order validation
    if result['status'] == 'no_change_needed':
        print(f"No order needed: {result['message']}")
        print(f"Current position: {result['current_position']} shares")
        print(f"Current allocation: {result['current_percent']*100:.2f}%")
        
    elif result['status'] == 'validated':
        action = result.get('action', 'UNKNOWN')
        quantity = result.get('quantity', 0)
        current_position = result.get('current_position', 0)
        print(f"Validated: Would {action} {quantity} shares of {symbol}")
        print(f"Current position: {current_position} shares")
        print(f"Portfolio value: ${result.get('portfolio_value', 0):,.2f}")
        
        # Now ask if we want to actually place the order
        confirmation = input("\nWould you like to place this order? (yes/no): ")
        if confirmation.lower() == 'yes':
            real_result = await app.order_target_percent(
                symbol=symbol,
                target_percent=target_percent,
                dry_run=False  # Actually place the order
            )
            
            if real_result['status'] == 'placed':
                print(f"✅ Order placed successfully! Order ID: {real_result.get('order_id')}")
            else:
                print(f"❌ Order failed: {real_result.get('error', 'Unknown error')}")
        else:
            print("Order cancelled by user")
            
    elif result['status'] == 'failed':
        print(f"❌ Order validation failed: {result.get('error', 'Unknown error')}")
    
    print("\n")


async def example_portfolio_rebalance(app: IBKRApp):
    """Demonstrate portfolio rebalance functionality"""
    print("\n=== Portfolio Rebalance Example ===")
    
    # Define target portfolio allocation
    # This is what you want your portfolio to look like
    target_portfolio = {
        "SPY": 0.30,   # 30% S&P 500
        "QQQ": 0.20,   # 20% Nasdaq
        "AAPL": 0.10,  # 10% Apple
        "MSFT": 0.10,  # 10% Microsoft
        "GOOGL": 0.10, # 10% Google
        "AMZN": 0.10,  # 10% Amazon
        # 10% cash
    }
    
    # First, get current allocations
    current = await app.get_portfolio_allocations()
    
    print("Current vs Target Allocations:")
    print(f"{'Symbol':<8} {'Current':<10} {'Target':<10} {'Diff':<10}")
    print("-" * 40)
    
    # Show all symbols from both current and target
    all_symbols = set(list(current.keys()) + list(target_portfolio.keys()))
    
    for symbol in sorted(all_symbols):
        current_pct = current.get(symbol, 0) * 100
        target_pct = target_portfolio.get(symbol, 0) * 100
        diff = target_pct - current_pct
        
        print(f"{symbol:<8} {current_pct:>7.2f}%   {target_pct:>7.2f}%   {diff:>+7.2f}%")
    
    # Get rebalance suggestions with 2% threshold
    suggestions = await app.suggest_rebalance(target_portfolio, threshold=0.02)
    
    needs_rebalance = suggestions.get('needs_rebalance', False)
    
    if needs_rebalance:
        print("\nRebalance Suggestions:")
        for symbol, data in suggestions.get('suggestions', {}).items():
            action = data['action']
            current = data['current_percent'] * 100
            target = data['target_percent'] * 100
            deviation = data['deviation'] * 100
            
            print(f"{symbol}: {action} from {current:.2f}% to {target:.2f}% (Deviation: {deviation:.2f}%)")
            
        # Ask if we want to rebalance
        confirmation = input("\nWould you like to rebalance the portfolio? (yes/no): ")
        if confirmation.lower() == 'yes':
            # Do a dry run first
            rebalance_result = await app.rebalance_portfolio(
                target_allocations=target_portfolio,
                dry_run=True
            )
            
            # Show what would happen
            summary = rebalance_result.get('summary', {})
            orders_count = summary.get('orders_placed', 0)
            print(f"\nRebalance would place {orders_count} orders")
            
            # Ask for final confirmation
            final_confirm = input("Proceed with actual rebalance? (yes/no): ")
            if final_confirm.lower() == 'yes':
                final_result = await app.rebalance_portfolio(
                    target_allocations=target_portfolio,
                    dry_run=False
                )
                print(f"\n✅ Portfolio rebalance initiated!")
                print(f"Orders placed: {final_result.get('summary', {}).get('orders_placed', 0)}")
            else:
                print("Rebalance cancelled by user")
        else:
            print("Rebalance cancelled by user")
    else:
        print("\nNo rebalance needed - portfolio is within threshold of target allocation")


async def main():
    """Main function"""
    # Initialize the application
    app = IBKRApp()
    
    try:
        # Start the application
        if not await app.start():
            logger.error("Failed to start trading application")
            return
        
        logger.info("IBKR application started successfully!")
        
        # Wait for connection to stabilize
        await asyncio.sleep(2)
        
        # Get overall portfolio summary
        portfolio = await app.get_portfolio_summary()
        net_liq = portfolio['account_summary']['net_liquidation']
        print(f"\nPortfolio Value: ${net_liq:,.2f}")
        
        # Show current allocations
        await show_portfolio_allocations(app)
        
        # Run the target percent order example
        choice = input("\nRun order_target_percent example? (yes/no): ")
        if choice.lower() == 'yes':
            await example_target_percent_order(app)
        
        # Run the portfolio rebalance example
        choice = input("\nRun portfolio rebalance example? (yes/no): ")
        if choice.lower() == 'yes':
            await example_portfolio_rebalance(app)
        
        print("\nExample completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in example: {e}")
    finally:
        # Always clean shutdown
        logger.info("Shutting down application...")
        await app.stop()
        logger.info("Application shutdown complete")


if __name__ == "__main__":
    print("=" * 60)
    print("Portfolio Allocation Example with order_target_percent")
    print("=" * 60)
    print("\nThis example demonstrates how to:")
    print("• Check current portfolio allocations")
    print("• Use order_target_percent to set target allocations")
    print("• Rebalance an entire portfolio to target allocations")
    print("\nWARNING: This will place REAL orders if confirmed!")
    print("Make sure you understand what the script will do before confirming.")
    print("\n")
    
    asyncio.run(main())
