# OrderManager Integration Guide

This guide explains how the OrderManager has been integrated into the main IBKR trading application and how to use it for automated trading.

## Overview

The OrderManager provides high-level order management capabilities that are now fully integrated into the main `IBKRApp` class. This integration gives you:

- **Unified Interface**: All trading functionality accessible through one main application class
- **Risk Management**: Built-in position sizing and risk controls
- **Order Tracking**: Automatic tracking of active and historical orders
- **Validation**: Pre-trade validation and dry-run capabilities
- **Clean Architecture**: Separation of concerns between data, analytics, and trading

## Architecture

```
IBKRApp (Main Application)
├── IBClient (Low-level IBKR API)
├── AsyncDataStorage (Database operations)
├── PortfolioAnalytics (Risk analysis)
└── OrderManager (High-level order management)
    └── Uses IBClient for actual order execution
```

## Key Features

### 1. High-Level Trading Methods

The main `IBKRApp` now provides these trading methods:

```python
# Basic order placement
await app.place_buy_order(symbol="AAPL", quantity=100, order_type="MARKET")
await app.place_sell_order(symbol="AAPL", quantity=50, order_type="LIMIT", limit_price=150.0)

# Portfolio allocation methods (NEW!)
await app.order_target_percent(symbol="AAPL", target_percent=0.10)  # 10% of portfolio
await app.order_percent(symbol="MSFT", percent=0.05)  # Buy 5% of portfolio

# Complete portfolio rebalancing
target_allocation = {"AAPL": 0.25, "MSFT": 0.20, "GOOGL": 0.15}
await app.rebalance_portfolio(target_allocation)

# Order management
await app.cancel_order(order_id=12345)
status = await app.get_order_status(order_id=12345)
active_orders = app.get_active_orders()
history = app.get_order_history(days=7)

# Market data and analysis
quote = await app.get_market_quote("AAPL")
allocations = await app.get_portfolio_allocations()
suggestions = await app.suggest_rebalance(target_allocation)

# Position management
await app.close_position("AAPL")  # Close entire position
```

### 2. Risk Management

Built-in risk controls include:

- **Position Size Limits**: Maximum position size per symbol
- **Order Value Limits**: Maximum single order value
- **Daily Loss Limits**: Stop trading if daily losses exceed threshold
- **Concentration Limits**: Prevent over-concentration in single positions

### 3. Order Types Supported

- `MARKET` - Market orders
- `LIMIT` - Limit orders  
- `STOP` - Stop orders
- `STOP_LIMIT` - Stop-limit orders
- `TRAIL` - Trailing stop orders
- `TRAIL_LIMIT` - Trailing stop-limit orders

### 4. Time in Force Options

- `DAY` - Good for day
- `GTC` - Good till canceled
- `IOC` - Immediate or cancel
- `FOK` - Fill or kill

## Basic Usage Example

```python
import asyncio
from src.app.main import IBKRApp

async def simple_trading_example():
    # Initialize the application
    app = IBKRApp()
    
    try:
        # Start the application (connects to IBKR)
        await app.start()
        
        # Get market quote
        quote = await app.get_market_quote("AAPL")
        print(f"AAPL: ${quote['last']:.2f}")
        
        # Place a dry-run order (validation only)
        result = await app.place_buy_order(
            symbol="AAPL",
            quantity=100,
            order_type="LIMIT",
            limit_price=150.00,
            dry_run=True  # Validate only, don't execute
        )
        
        if result['status'] == 'validated':
            print("Order validation successful!")
            
            # Actually place the order (remove dry_run=True)
            # result = await app.place_buy_order(
            #     symbol="AAPL",
            #     quantity=100,
            #     order_type="LIMIT", 
            #     limit_price=150.00
            # )
        
        # Check active orders
        active_orders = app.get_active_orders()
        print(f"Active orders: {len(active_orders)}")
        
    finally:
        await app.stop()

# Run the example
asyncio.run(simple_trading_example())
```

## Advanced Features

### 1. Portfolio Integration

The OrderManager integrates with portfolio analytics:

```python
# Get comprehensive portfolio summary including positions
portfolio = await app.get_portfolio_summary()

# Generate risk report
risk_report = await app.get_risk_report(days=30)

# Check if order violates risk limits before placing
result = await app.place_buy_order(
    symbol="AAPL", 
    quantity=1000,  # Large order
    order_type="MARKET",
    dry_run=True  # Check first
)
```

### 2. Background Processing

The application runs background tasks that:

- Monitor connection health
- Update portfolio metrics
- Clean up old data
- Track order status changes

### 3. Market Data Streaming

Integrated market data capabilities:

```python
# Start streaming market data
await app.start_market_data_stream("AAPL")

# Data is automatically stored in database
# Stop streaming when done
await app.stop_market_data_stream("AAPL")
```

## Configuration

The OrderManager can be configured through the main application:

```python
# Access the order manager directly for advanced configuration
app.order_manager.max_position_size = 5000
app.order_manager.max_order_value = 100000
app.order_manager.daily_loss_limit = 10000
```

## Error Handling

The system provides comprehensive error handling:

```python
try:
    result = await app.place_buy_order(
        symbol="INVALID",
        quantity=100,
        order_type="MARKET"
    )
    
    if result['status'] == 'failed':
        print(f"Order failed: {result['error']}")
    elif result['status'] == 'placed':
        print(f"Order placed successfully: {result['order_id']}")
        
except ConnectionError:
    print("Not connected to IBKR")
except ValidationError as e:
    print(f"Order validation failed: {e}")
```

## Safety Features

### 1. Dry Run Mode

Always test orders first:

```python
# Test the order first
result = await app.place_buy_order(..., dry_run=True)
if result['status'] == 'validated':
    # Now place the real order
    result = await app.place_buy_order(..., dry_run=False)
```

### 2. Position Limits

The system prevents excessive position sizes:

```python
# This will be rejected if it exceeds position limits
result = await app.place_buy_order(
    symbol="AAPL",
    quantity=10000,  # Very large order
    order_type="MARKET"
)
```

### 3. Risk Monitoring

Continuous risk monitoring:

```python
# Check current risk exposure
risk_report = await app.get_risk_report()
alerts = risk_report.get('risk_alerts', [])

for alert in alerts:
    print(f"{alert['severity']}: {alert['message']}")
```

## Portfolio Allocation Features (NEW!)

The integrated system now includes powerful portfolio allocation methods that were in the original `client.py` file:

### order_target_percent

The key portfolio allocation function - automatically calculates whether to buy or sell to reach a target allocation:

```python
# Set AAPL to 10% of portfolio value
result = await app.order_target_percent(
    symbol="AAPL", 
    target_percent=0.10,  # 10%
    dry_run=True  # Validate first
)

# The function automatically:
# - Gets current portfolio value
# - Calculates current position value
# - Determines if we need to buy or sell
# - Places the appropriate order
```

### order_percent

Place an order for a percentage of portfolio value (without considering existing positions):

```python
# Buy $1000 worth if portfolio is $10,000
result = await app.order_percent(
    symbol="MSFT",
    percent=0.10,  # 10% of portfolio
    order_type="MARKET"
)
```

### rebalance_portfolio

Rebalance entire portfolio to target allocations in one operation:

```python
# Define your ideal portfolio
target_allocation = {
    "AAPL": 0.25,   # 25%
    "MSFT": 0.20,   # 20%
    "GOOGL": 0.15,  # 15%
    "AMZN": 0.10,   # 10%
    "TSLA": 0.10,   # 10%
    # 20% remains in cash
}

# Rebalance entire portfolio
result = await app.rebalance_portfolio(
    target_allocations=target_allocation,
    dry_run=True  # Test first
)

# Shows what orders would be placed
print(f"Would place {result['summary']['orders_placed']} orders")
```

### get_portfolio_allocations

See current portfolio allocations:

```python
allocations = await app.get_portfolio_allocations()
for symbol, percent in allocations.items():
    print(f"{symbol}: {percent*100:.2f}%")
```

### suggest_rebalance

Get suggestions for rebalancing based on deviation thresholds:

```python
suggestions = await app.suggest_rebalance(
    target_allocations=target_allocation,
    threshold=0.05  # 5% deviation threshold
)

if suggestions['needs_rebalance']:
    print("Rebalance recommended:")
    for symbol, data in suggestions['suggestions'].items():
        print(f"{symbol}: {data['action']} from {data['current_percent']*100:.1f}% to {data['target_percent']*100:.1f}%")
```

### Complete Portfolio Management Workflow

```python
async def manage_portfolio(app):
    # 1. Check current allocations
    current = await app.get_portfolio_allocations()
    print("Current allocations:", current)
    
    # 2. Define target portfolio
    target = {
        "SPY": 0.40,   # 40% S&P 500
        "VTI": 0.30,   # 30% Total Stock Market
        "BND": 0.20,   # 20% Bonds
        "VEA": 0.10,   # 10% International
    }
    
    # 3. Get rebalance suggestions
    suggestions = await app.suggest_rebalance(target, threshold=0.05)
    
    # 4. If rebalance needed, do it
    if suggestions['needs_rebalance']:
        result = await app.rebalance_portfolio(target, dry_run=False)
        print(f"Rebalanced! Orders placed: {result['summary']['orders_placed']}")
    else:
        print("Portfolio is already balanced")
```

## Examples

See the `examples/` directory for complete working examples:

- `examples/trading_example.py` - Basic trading operations
- `examples/simple_strategy.py` - Complete trading strategy
- `examples/portfolio_allocation_example.py` - Portfolio allocation and rebalancing

## Best Practices

1. **Always use dry_run first**: Test orders before placing them live
2. **Monitor risk**: Regularly check risk reports and position sizes
3. **Handle errors**: Implement proper error handling for all trading operations
4. **Clean shutdown**: Always call `await app.stop()` when done
5. **Log everything**: Enable logging to track all trading activities

## Troubleshooting

### Common Issues

1. **Connection Errors**: Ensure IBKR Gateway/TWS is running and properly configured
2. **Order Validation Failures**: Check symbol, quantity, and price parameters
3. **Risk Limit Violations**: Review position sizes and risk parameters
4. **Missing Market Data**: Ensure you have market data subscriptions

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

This integration provides a complete, production-ready trading system with proper risk management, error handling, and monitoring capabilities.
