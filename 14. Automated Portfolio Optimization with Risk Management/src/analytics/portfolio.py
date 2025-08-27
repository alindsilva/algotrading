"""
Portfolio analytics module for risk management and performance metrics.
Provides asynchronous calculation of key risk and performance indicators.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

from ..core.types import ReturnSeries, PositionData, AccountValue
from ..core.exceptions import ValidationError, CalculationError

logger = logging.getLogger(__name__)


@dataclass
class PortfolioMetrics:
    """Collection of portfolio performance and risk metrics"""
    
    # Performance metrics
    total_return: float
    annualized_return: float
    cumulative_returns: pd.Series
    
    # Risk metrics
    volatility: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float
    
    # Tail risk metrics
    var_95: float
    cvar_95: float
    
    # Correlation metrics
    beta_to_market: Optional[float] = None
    correlation_to_market: Optional[float] = None
    
    # Additional info
    calculation_time: datetime = None
    time_period_days: int = None
    
    def __post_init__(self):
        if self.calculation_time is None:
            self.calculation_time = datetime.now()


class PortfolioAnalytics:
    """
    Provides asynchronous portfolio analytics and risk metrics calculation.
    """
    
    def __init__(self, 
                risk_free_rate: float = 0.0, 
                benchmark_returns: Optional[pd.Series] = None,
                window_size: int = 252):
        """
        Initialize portfolio analytics
        
        Args:
            risk_free_rate: Annualized risk-free rate (default 0)
            benchmark_returns: Optional benchmark return series
            window_size: Number of periods for rolling calculations (default 252 - trading days in year)
        """
        self.risk_free_rate = risk_free_rate
        self.benchmark_returns = benchmark_returns
        self.window_size = window_size
        
        # Daily risk-free rate derived from annual rate
        self._daily_risk_free_rate = (1 + risk_free_rate) ** (1 / 252) - 1
    
    async def calculate_portfolio_metrics(self, 
                                        returns: Union[pd.Series, np.ndarray], 
                                        prices: Optional[pd.Series] = None) -> PortfolioMetrics:
        """
        Calculate comprehensive portfolio metrics asynchronously
        
        Args:
            returns: Return series (daily returns preferred)
            prices: Optional price series (will be used to calculate returns if provided)
        
        Returns:
            PortfolioMetrics object with calculated metrics
        """
        # Input validation and conversion
        if prices is not None and returns is None:
            returns = await self._calculate_returns_from_prices(prices)
        
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)
        
        if not isinstance(returns, pd.Series):
            raise ValidationError("Returns must be a pandas Series or numpy array")
        
        if len(returns) < 5:
            raise ValidationError("Insufficient data for portfolio metrics calculation")
        
        # Calculate metrics concurrently
        tasks = [
            self._calculate_performance_metrics(returns),
            self._calculate_risk_metrics(returns),
            self._calculate_tail_risk_metrics(returns),
            self._calculate_correlation_metrics(returns) if self.benchmark_returns is not None else None
        ]
        
        # Filter out None tasks
        tasks = [t for t in tasks if t is not None]
        
        # Execute all calculations concurrently
        results = await asyncio.gather(*tasks)
        
        # Unpack results
        performance_metrics = results[0]
        risk_metrics = results[1]
        tail_risk_metrics = results[2]
        correlation_metrics = results[3] if len(results) > 3 else None
        
        # Combine all metrics
        metrics = PortfolioMetrics(
            # Performance metrics
            total_return=performance_metrics['total_return'],
            annualized_return=performance_metrics['annualized_return'],
            cumulative_returns=performance_metrics['cumulative_returns'],
            
            # Risk metrics
            volatility=risk_metrics['volatility'],
            max_drawdown=risk_metrics['max_drawdown'],
            sharpe_ratio=risk_metrics['sharpe_ratio'],
            sortino_ratio=risk_metrics['sortino_ratio'],
            calmar_ratio=risk_metrics['calmar_ratio'],
            omega_ratio=risk_metrics['omega_ratio'],
            
            # Tail risk metrics
            var_95=tail_risk_metrics['var_95'],
            cvar_95=tail_risk_metrics['cvar_95'],
            
            # Correlation metrics (if available)
            beta_to_market=correlation_metrics['beta'] if correlation_metrics else None,
            correlation_to_market=correlation_metrics['correlation'] if correlation_metrics else None,
            
            # Additional info
            time_period_days=len(returns)
        )
        
        return metrics
    
    async def _calculate_returns_from_prices(self, prices: pd.Series) -> pd.Series:
        """Calculate returns from price series"""
        if not isinstance(prices, pd.Series):
            raise ValidationError("Prices must be a pandas Series")
        
        if len(prices) < 2:
            raise ValidationError("Insufficient price data to calculate returns")
        
        # Calculate simple returns (pt / pt-1 - 1)
        returns = prices.pct_change().dropna()
        
        return returns
    
    async def _calculate_performance_metrics(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate performance metrics"""
        # Cumulative returns
        cumulative_returns = (1 + returns).cumprod()
        
        # Total return
        total_return = cumulative_returns.iloc[-1] - 1
        
        # Annualized return (based on 252 trading days)
        n_periods = len(returns)
        annualized_return = (1 + total_return) ** (252 / n_periods) - 1
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'cumulative_returns': cumulative_returns
        }
    
    async def _calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate risk metrics"""
        # Volatility (annualized)
        volatility = returns.std() * np.sqrt(252)
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.cummax()
        drawdowns = (cumulative_returns / rolling_max) - 1
        max_drawdown = abs(drawdowns.min())
        
        # Sharpe ratio
        excess_returns = returns - self._daily_risk_free_rate
        sharpe_ratio = (excess_returns.mean() / returns.std()) * np.sqrt(252)
        
        # Sortino ratio (downside risk)
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252)
        sortino_ratio = 0.0 if downside_deviation == 0 else (excess_returns.mean() / downside_deviation) * np.sqrt(252)
        
        # Calmar ratio
        calmar_ratio = 0.0 if max_drawdown == 0 else (returns.mean() * 252) / max_drawdown
        
        # Omega ratio
        threshold = self._daily_risk_free_rate
        omega_ratio = self._calculate_omega_ratio(returns, threshold)
        
        return {
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'omega_ratio': omega_ratio
        }
    
    def _calculate_omega_ratio(self, returns: pd.Series, threshold: float) -> float:
        """Calculate omega ratio"""
        excess_returns = returns - threshold
        positive_sum = excess_returns[excess_returns > 0].sum()
        negative_sum = abs(excess_returns[excess_returns < 0].sum())
        
        if negative_sum == 0:
            return float('inf')  # If no negative returns, omega is infinite
        
        return positive_sum / negative_sum
    
    async def _calculate_tail_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate tail risk metrics"""
        # Value at Risk (VaR)
        var_95 = abs(np.percentile(returns, 5))
        
        # Conditional Value at Risk (CVaR)
        cvar_95 = abs(returns[returns <= -var_95].mean())
        
        return {
            'var_95': var_95,
            'cvar_95': cvar_95
        }
    
    async def _calculate_correlation_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate correlation and beta to benchmark"""
        if self.benchmark_returns is None:
            return {'beta': None, 'correlation': None}
        
        # Align return series
        common_index = returns.index.intersection(self.benchmark_returns.index)
        if len(common_index) < 5:
            logger.warning("Insufficient overlapping data for correlation metrics")
            return {'beta': None, 'correlation': None}
        
        portfolio_returns = returns.loc[common_index]
        benchmark_returns = self.benchmark_returns.loc[common_index]
        
        # Correlation
        correlation = portfolio_returns.corr(benchmark_returns)
        
        # Beta
        covariance = portfolio_returns.cov(benchmark_returns)
        benchmark_variance = benchmark_returns.var()
        beta = 0.0 if benchmark_variance == 0 else covariance / benchmark_variance
        
        return {
            'beta': beta,
            'correlation': correlation
        }
    
    async def analyze_positions(self, 
                              positions: List[PositionData], 
                              account_values: List[AccountValue]) -> Dict[str, Any]:
        """
        Analyze current positions and account values
        
        Args:
            positions: List of position data objects
            account_values: List of account value objects
        
        Returns:
            Dictionary with position analytics
        """
        if not positions:
            return {
                'total_positions': 0,
                'total_value': 0.0,
                'long_exposure': 0.0,
                'short_exposure': 0.0,
                'net_exposure': 0.0,
                'gross_exposure': 0.0,
                'positions_by_type': {},
                'largest_positions': []
            }
        
        # Get account values needed for calculations
        total_equity = self._get_account_value(account_values, 'NetLiquidation')
        
        # Analyze positions
        positions_by_type = {}
        long_exposure = 0.0
        short_exposure = 0.0
        
        # Process positions
        position_summaries = []
        for pos in positions:
            position_value = pos.position * pos.avg_cost
            
            # Track by security type
            if pos.sec_type not in positions_by_type:
                positions_by_type[pos.sec_type] = {'count': 0, 'value': 0.0}
            
            positions_by_type[pos.sec_type]['count'] += 1
            positions_by_type[pos.sec_type]['value'] += abs(position_value)
            
            # Track exposure
            if pos.position > 0:
                long_exposure += position_value
            else:
                short_exposure += position_value  # This is negative for short positions
            
            # Record position summary
            position_summaries.append({
                'symbol': pos.symbol,
                'sec_type': pos.sec_type,
                'position': pos.position,
                'avg_cost': pos.avg_cost,
                'value': position_value,
                'pct_of_portfolio': (abs(position_value) / total_equity) if total_equity else 0.0
            })
        
        # Sort positions by absolute value
        largest_positions = sorted(position_summaries, 
                                 key=lambda x: abs(x['value']), 
                                 reverse=True)[:10]  # Top 10
        
        # Calculate exposures
        gross_exposure = long_exposure + abs(short_exposure)
        net_exposure = long_exposure - abs(short_exposure)
        
        return {
            'total_positions': len(positions),
            'total_value': gross_exposure,
            'long_exposure': long_exposure,
            'short_exposure': short_exposure,
            'net_exposure': net_exposure,
            'gross_exposure': gross_exposure,
            'net_exposure_pct': (net_exposure / total_equity) if total_equity else 0.0,
            'gross_exposure_pct': (gross_exposure / total_equity) if total_equity else 0.0,
            'positions_by_type': positions_by_type,
            'largest_positions': largest_positions
        }
    
    def _get_account_value(self, account_values: List[AccountValue], key: str) -> float:
        """Extract specific account value by key"""
        for value in account_values:
            if value.key == key and value.currency == 'USD':
                try:
                    return float(value.value)
                except (ValueError, TypeError):
                    return 0.0
        return 0.0
    
    async def calculate_rolling_metrics(self, 
                                      returns: pd.Series, 
                                      window: int = None) -> Dict[str, pd.Series]:
        """
        Calculate rolling risk metrics
        
        Args:
            returns: Return series
            window: Rolling window size (defaults to self.window_size)
        
        Returns:
            Dictionary of rolling metrics series
        """
        if window is None:
            window = self.window_size
        
        if len(returns) < window:
            raise ValidationError(f"Insufficient return data for {window}-period rolling metrics")
        
        # Initialize rolling metric series
        rolling_volatility = pd.Series(index=returns.index)
        rolling_sharpe = pd.Series(index=returns.index)
        rolling_sortino = pd.Series(index=returns.index)
        rolling_max_drawdown = pd.Series(index=returns.index)
        
        # Calculate metrics for each window
        for i in range(window, len(returns) + 1):
            window_returns = returns.iloc[i - window:i]
            
            # Volatility
            vol = window_returns.std() * np.sqrt(252)
            rolling_volatility.iloc[i - 1] = vol
            
            # Sharpe ratio
            excess_return = window_returns.mean() - self._daily_risk_free_rate
            sharpe = (excess_return / window_returns.std()) * np.sqrt(252)
            rolling_sharpe.iloc[i - 1] = sharpe
            
            # Sortino ratio
            downside_returns = window_returns[window_returns < 0]
            if len(downside_returns) > 0:
                downside_deviation = downside_returns.std() * np.sqrt(252)
                if downside_deviation > 0:
                    sortino = (excess_return / downside_deviation) * np.sqrt(252)
                    rolling_sortino.iloc[i - 1] = sortino
            
            # Max drawdown
            cum_returns = (1 + window_returns).cumprod()
            rolling_max = cum_returns.cummax()
            drawdowns = (cum_returns / rolling_max) - 1
            max_dd = abs(drawdowns.min())
            rolling_max_drawdown.iloc[i - 1] = max_dd
        
        return {
            'rolling_volatility': rolling_volatility.dropna(),
            'rolling_sharpe_ratio': rolling_sharpe.dropna(),
            'rolling_sortino_ratio': rolling_sortino.dropna(),
            'rolling_max_drawdown': rolling_max_drawdown.dropna()
        }
    
    async def stress_test_portfolio(self, 
                                  returns: pd.Series, 
                                  scenarios: Dict[str, float] = None) -> Dict[str, float]:
        """
        Perform stress testing on portfolio returns
        
        Args:
            returns: Historical return series
            scenarios: Dictionary of scenario descriptions and shock magnitudes
                      (e.g., {'market_crash': -0.15, 'interest_rate_spike': -0.05})
        
        Returns:
            Dictionary of scenario impacts
        """
        if scenarios is None:
            # Default scenarios if none provided
            scenarios = {
                'market_crash_15pct': -0.15,
                'market_crash_10pct': -0.10,
                'market_crash_5pct': -0.05,
                'interest_rate_increase': -0.03,
                'volatility_spike': -0.07,
            }
        
        # Calculate baseline metrics
        baseline_metrics = await self._calculate_risk_metrics(returns)
        
        # Calculate stressed metrics for each scenario
        stress_results = {}
        
        for scenario_name, shock in scenarios.items():
            # Apply stress shock to returns
            stressed_returns = returns.copy()
            
            # Apply shock to most recent returns (last 5 days)
            stress_window = min(5, len(stressed_returns))
            stressed_returns.iloc[-stress_window:] = stressed_returns.iloc[-stress_window:] + shock
            
            # Calculate stressed metrics
            stressed_metrics = await self._calculate_risk_metrics(stressed_returns)
            
            # Calculate impact
            impact = {
                'new_sharpe_ratio': stressed_metrics['sharpe_ratio'],
                'sharpe_ratio_change': stressed_metrics['sharpe_ratio'] - baseline_metrics['sharpe_ratio'],
                'new_max_drawdown': stressed_metrics['max_drawdown'],
                'max_drawdown_change': stressed_metrics['max_drawdown'] - baseline_metrics['max_drawdown'],
                'new_volatility': stressed_metrics['volatility'],
                'volatility_change': stressed_metrics['volatility'] - baseline_metrics['volatility']
            }
            
            stress_results[scenario_name] = impact
        
        return stress_results
