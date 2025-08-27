#!/usr/bin/env python3
"""
Example script demonstrating how to use the integrated IBKR trading application.

This example shows how to:
1. Start the trading application
2. Get portfolio and market data
3. Place orders (buy/sell)
4. Monitor order status
5. Generate risk reports
6. Clean shutdown

Run this example after setting up your IBKR configuration.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path

# Import the main trading application
from src.app.main import IBKRApp

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Main example function demonstrating the trading application"""
    
    # Initialize the trading application
    app = IBKRApp()  # You can pass config_path="path/to/config.json" if you have one
    
    try:
        logger.info("Starting IBKR Trading Application Example...")
        
        # Start the application (connects to IBKR, initializes all components)
        if not await app.start():
            logger.error("Failed to start trading application")
            return
        
        logger.info("✅ Application started successfully!")
        
        # Wait a moment for connection to stabilize
        await asyncio.sleep(2)
        
        # Example 1: Get application status
        logger.info("=== Application Status ===")
        status = app.status
        print(f"Running: {status['running']}")
        print(f"Connected: {status['connected']}")
        print(f"Active Orders: {status['active_orders']}")
        print()
        
        # Example 2: Get portfolio summary
        logger.info("=== Portfolio Summary ===")
        try:
            portfolio = await app.get_portfolio_summary()
            print(f"Portfolio Value: ${portfolio['account_summary']['net_liquidation']:,.2f}")
            print(f"Cash Available: ${portfolio['account_summary']['total_cash']:,.2f}")
            print(f"Buying Power: ${portfolio['account_summary']['buying_power']:,.2f}")
            
            if portfolio['positions'] and len(portfolio['positions']['positions']) > 0:
                print(f"Number of Positions: {len(portfolio['positions']['positions'])}")
                for pos in portfolio['positions']['positions'][:3]:  # Show first 3
                    print(f"  {pos['symbol']}: {pos['quantity']} shares @ ${pos['avg_cost']:.2f}")
            else:
                print("No current positions")
            print()
        except Exception as e:
            logger.error(f"Failed to get portfolio summary: {e}")
        
        # Example 3: Get market quote
        logger.info("=== Market Quote Example ===")
        try:
            quote = await app.get_market_quote("AAPL")
            if quote:
                print(f"AAPL Quote:")
                print(f"  Bid: ${quote['bid']:.2f}")
                print(f"  Ask: ${quote['ask']:.2f}")
                print(f"  Last: ${quote['last']:.2f}")
                print(f"  Volume: {quote['volume']:,}")
            else:
                print("Could not retrieve AAPL quote")
            print()
        except Exception as e:
            logger.error(f"Failed to get market quote: {e}")
        
        # Example 4: Place a dry-run buy order (validation only)
        logger.info("=== Order Placement Example (Dry Run) ===")
        try:
            # This validates the order but doesn't actually place it
            buy_result = await app.place_buy_order(
                symbol="AAPL",
                quantity=10,
                order_type="LIMIT",
                limit_price=150.00,
                dry_run=True  # This means validate only, don't actually place
            )
            print(f"Buy Order Validation Result: {buy_result['status']}")
            if buy_result['status'] == 'validated':
                print("✅ Order validation successful - order parameters are valid")
            else:
                print(f"❌ Order validation failed: {buy_result.get('error', 'Unknown error')}")
            print()
        except Exception as e:
            logger.error(f"Failed to validate buy order: {e}")
        
        # Example 5: Check active orders
        logger.info("=== Active Orders ===")
        try:
            active_orders = app.get_active_orders()
            if active_orders:
                print(f"Active Orders: {len(active_orders)}")
                for order in active_orders:
                    print(f"  Order {order['order_id']}: {order['action']} {order['quantity']} "
                          f"{order['symbol']} @ {order['order_type']} - Status: {order['status']}")
            else:
                print("No active orders")
            print()
        except Exception as e:
            logger.error(f"Failed to get active orders: {e}")
        
        # Example 6: Get order history
        logger.info("=== Order History (Last 7 Days) ===")
        try:
            order_history = app.get_order_history(days=7)
            if order_history:
                print(f"Recent Orders: {len(order_history)}")
                for order in order_history[-3:]:  # Show last 3
                    print(f"  {order['created_at'][:19]}: {order['action']} {order['quantity']} "
                          f"{order['symbol']} - Status: {order['status']}")
            else:
                print("No recent order history")
            print()
        except Exception as e:
            logger.error(f"Failed to get order history: {e}")
        
        # Example 7: Generate risk report
        logger.info("=== Risk Report ===")
        try:
            risk_report = await app.get_risk_report(days=30)
            
            if 'error' in risk_report:
                print(f"Risk analysis unavailable: {risk_report['error']}")
            else:
                metrics = risk_report['portfolio_metrics']
                print(f"30-Day Risk Analysis:")
                print(f"  Total Return: {metrics['total_return']:.2%}")
                print(f"  Annualized Return: {metrics['annualized_return']:.2%}")
                print(f"  Volatility: {metrics['volatility']:.2%}")
                print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
                print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
                print(f"  VaR (95%): {metrics['var_95']:.2%}")
                
                # Show risk alerts
                alerts = risk_report['risk_alerts']
                if alerts:
                    print(f"  Risk Alerts:")
                    for alert in alerts:
                        print(f"    {alert['severity']}: {alert['message']}")
                else:
                    print("  No risk alerts")
            print()
        except Exception as e:
            logger.error(f"Failed to generate risk report: {e}")
        
        # Example 8: Start market data streaming (optional)
        logger.info("=== Market Data Streaming Example ===")
        try:
            # Start streaming for AAPL (this will run in background)
            streaming_started = await app.start_market_data_stream("AAPL")
            if streaming_started:
                print("✅ Started market data streaming for AAPL")
                print("Market data will be stored in the database automatically")
                
                # Let it stream for a few seconds
                await asyncio.sleep(5)
                
                # Stop the streaming
                await app.stop_market_data_stream("AAPL")
                print("✅ Stopped market data streaming for AAPL")
            else:
                print("❌ Failed to start market data streaming")
            print()
        except Exception as e:
            logger.error(f"Failed to manage market data streaming: {e}")
        
        # Example 9: Demonstrate position management (dry run)
        logger.info("=== Position Management Example ===")
        try:
            # This would close any existing AAPL position (dry run mode)
            close_result = await app.close_position("AAPL", dry_run=True)
            print(f"Close Position Result: {close_result['status']}")
            
            if close_result['status'] == 'no_position':
                print("No AAPL position to close")
            elif close_result['status'] == 'validated':
                original_pos = close_result.get('original_position', 0)
                print(f"Would close position of {original_pos} shares")
            elif close_result['status'] == 'failed':
                print(f"Error: {close_result.get('error', 'Unknown error')}")
            print()
        except Exception as e:
            logger.error(f"Failed to check position closure: {e}")
        
        # Keep the application running for a bit to demonstrate background tasks
        logger.info("=== Background Tasks Demo ===")
        print("Application will run for 30 seconds to demonstrate background tasks...")
        print("(Portfolio metrics updates, connection monitoring, etc.)")
        
        # Wait and show periodic status updates
        for i in range(6):
            await asyncio.sleep(5)
            status = app.status
            print(f"Status check {i+1}: Connected={status['connected']}, "
                  f"Background tasks={status['background_tasks']}")
        
        logger.info("Example completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        # Always clean shutdown
        logger.info("Shutting down application...")
        await app.stop()
        logger.info("✅ Application shutdown complete")


def run_example():
    """Run the trading example"""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExample interrupted by user")
    except Exception as e:
        print(f"Example failed: {e}")


if __name__ == "__main__":
    print("IBKR Trading Application Integration Example")
    print("=" * 50)
    print()
    print("This example demonstrates:")
    print("• Application startup and connection")
    print("• Portfolio and account data retrieval")
    print("• Market data quotes")
    print("• Order placement and management")
    print("• Risk reporting and analysis")
    print("• Market data streaming")
    print("• Clean application shutdown")
    print()
    print("Note: This example uses dry-run mode for orders")
    print("      No actual trades will be executed")
    print()
    input("Press Enter to continue...")
    print()
    
    run_example()
