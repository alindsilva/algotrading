Usage
Configure Portfolio: Define your portfolio and set the desired rebalancing parameters.
Run Optimization: Execute the optimization script to calculate the optimal weights.
Apply Weights: Use the integration with Interactive Brokers TWS API to apply the calculated weights to your portfolio.

# Example code to demonstrate usage
from riskfolio import Portfolio
from ib_insync import IB

# Define your portfolio
portfolio = Portfolio(assets=['AAPL', 'MSFT', 'GOOGL'])

# Calculate optimal weights
weights = portfolio.optimize()

# Connect to Interactive Brokers TWS API
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

# Apply weights to portfolio
for asset, weight in weights.items():
    # Code to place orders based on calculated weights
    pass

