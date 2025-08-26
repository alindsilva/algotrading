# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This repository contains the code for the "Python for Algorithmic Trading Cookbook" by Packt Publishing. It provides practical recipes for designing, building, and deploying algorithmic trading strategies using Python, covering the complete workflow from data acquisition to live trading.

## Repository Structure

The codebase is organized into numbered chapters, each focusing on specific aspects of algorithmic trading:

- **01-04**: Data acquisition, analysis, visualization, and storage (foundations)
- **05**: Alpha factor development using machine learning and statistical methods
- **06**: Vector-based backtesting with VectorBT for strategy optimization
- **07**: Event-driven backtesting with Zipline Reloaded for production-ready testing
- **08-09**: Performance evaluation using AlphaLens and Pyfolio
- **10-13**: Interactive Brokers API integration for live trading
- **14**: Automated portfolio optimization with risk management

## Key Technologies and Frameworks

### Core Data Stack
- **OpenBB Platform**: Primary data source for market data (equities, futures, options)
- **pandas**: Data manipulation and time series analysis
- **NumPy**: Numerical computations and array operations
- **SQLite/PostgreSQL**: Data storage and persistence

### Backtesting and Strategy Development
- **Zipline Reloaded**: Event-driven backtesting framework for factor strategies
- **VectorBT**: High-performance vectorized backtesting for parameter optimization
- **AlphaLens**: Factor analysis and performance evaluation
- **Pyfolio**: Portfolio and risk metrics analysis

### Live Trading Infrastructure
- **Interactive Brokers API (ibapi)**: Broker connectivity for live trading
- **Threading**: Asynchronous order management and data streaming

### Visualization and Analysis
- **Matplotlib/Seaborn**: Static plotting and statistical visualizations
- **Plotly Dash**: Interactive web applications for factor analysis

## Common Development Commands

### Environment Setup
Since there are no standard dependency files, you'll need to install packages as needed:
```bash
# Core data libraries
pip install pandas numpy openbb matplotlib seaborn plotly

# Backtesting frameworks  
pip install zipline-reloaded vectorbt alphalens pyfolio

# Interactive Brokers API
pip install ibapi

# Additional analysis tools
pip install scikit-learn empyrical logbook
```

### Running Jupyter Notebooks
Most examples are in Jupyter notebooks:
```bash
jupyter notebook  # Start notebook server
```

### Zipline Bundle Management
For Zipline Reloaded backtesting:
```bash
# Ingest data bundle (may need custom bundle setup)
zipline ingest -b daily_us_equities

# Check available bundles
zipline bundles
```

### Interactive Brokers Connection
For live trading examples (Chapter 10+):
```bash
# Start TWS or IB Gateway on localhost:7497 (paper trading)
# Then run Python scripts that connect to IB API
python app.py
```

### Running Dash Applications
For interactive web apps:
```bash
python app.py  # Usually runs on http://127.0.0.1:8050
```

## Architecture Patterns

### Data Pipeline Architecture
1. **Data Acquisition**: OpenBB Platform fetches data from multiple providers
2. **Data Storage**: Cleaned data stored in SQLite/PostgreSQL with proper indexing
3. **Factor Engineering**: Custom factors built using pandas and pipeline frameworks
4. **Backtesting**: Event-driven simulation with Zipline or vectorized with VectorBT
5. **Performance Analysis**: Risk-adjusted metrics via AlphaLens/Pyfolio
6. **Live Deployment**: IB API integration for automated trading

### Zipline Pipeline System
The repository extensively uses Zipline's Pipeline API for factor development:
- **Custom Factors**: Inherit from `CustomFactor` and implement `compute()` method
- **Pipeline Construction**: Combine factors, filters, and ranking logic
- **Alpha Discovery**: Screen universes and rank securities by factor exposure

### Interactive Brokers Integration Pattern
Live trading follows a consistent pattern across chapters:
- **Wrapper Class**: Inherits from `EWrapper` to handle IB API callbacks
- **Client Class**: Inherits from `EClient` for sending requests to IB
- **App Class**: Combines wrapper and client with threading for async operations
- **Contract/Order Abstraction**: Helper functions for creating financial instruments and orders

### Factor Development Workflow
1. **Factor Definition**: Create custom factors using historical price/volume data
2. **Universe Selection**: Filter securities by liquidity/market cap criteria  
3. **Ranking and Selection**: Rank securities by factor values, select top/bottom
4. **Portfolio Construction**: Equal-weight or risk-weighted portfolio allocation
5. **Rebalancing Logic**: Periodic rebalancing with transaction cost considerations

## Key Implementation Details

### OpenBB Platform Usage
```python
from openbb import obb
obb.user.preferences.output_type = "dataframe"  # Always set for pandas compatibility
data = obb.equity.price.historical("AAPL", provider="yfinance")
```

### Zipline Factor Pattern
```python
class CustomFactor(zipline.pipeline.CustomFactor):
    inputs = [USEquityPricing.close]
    window_length = 252
    
    def compute(self, today, assets, out, prices):
        # Factor logic here - always populate out[:]
        out[:] = factor_values
```

### IB API Connection Pattern
```python
class IBApp(IBWrapper, IBClient):
    def __init__(self, ip, port, client_id):
        IBWrapper.__init__(self)
        IBClient.__init__(self, wrapper=self)
        self.connect(ip, port, client_id)
        
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()
```

## Database and File Management

### Data Storage Locations
- **SQLite databases**: Used for tick data storage (`tick_data.sqlite`)
- **Pickle files**: Serialized backtest results and factor data
- **CSV exclusions**: Heavy data files excluded via `.gitignore`
- **Log files**: IB API logs stored in `log/` directory

### Common Data Patterns
- Time series data indexed by date with symbol columns
- Factor data stored as pandas DataFrames with asset identifiers
- Portfolio data tracked with positions, cash, and performance metrics

## Testing and Validation

### Backtesting Validation
- **Statistical Significance**: Use multiple time periods and out-of-sample testing
- **Transaction Costs**: Include realistic commission and slippage models
- **Survivorship Bias**: Account for delisted securities in historical analysis
- **Market Regime Changes**: Test strategies across different market conditions

### Production Readiness
- **Error Handling**: Robust exception handling for market data failures
- **Logging**: Comprehensive logging for debugging and audit trails  
- **Risk Management**: Position sizing and portfolio-level risk controls
- **Monitoring**: Real-time performance tracking and alerts

## Development Workflow Notes

### Jupyter Notebook Best Practices
- Notebooks are organized by recipe/chapter with descriptive naming
- Each notebook is self-contained with imports and setup
- Data dependencies clearly documented at the beginning
- Results and visualizations included for reference

### Code Reusability
- Common functionality abstracted into helper modules (contracts, orders, utils)
- Consistent patterns across different asset classes and strategies
- Modular design allows mixing and matching components

### Data Dependencies
- External data sources require API keys or subscriptions
- Some examples may need historical data ingestion before running
- Interactive Brokers examples require active TWS/Gateway connection

## Troubleshooting Common Issues

### Zipline Bundle Issues
- Custom data bundles may need manual setup
- Bundle ingestion can fail with data type errors
- Graphviz required for pipeline visualization (`brew install graphviz`)

### Interactive Brokers Connection
- TWS/Gateway must be configured to accept API connections
- Client ID conflicts can prevent connection
- Market data subscriptions may be required for live data

### OpenBB Platform
- Some providers require API keys in environment variables
- Rate limiting may affect data retrieval speed
- Provider availability varies by asset type and timeframe
