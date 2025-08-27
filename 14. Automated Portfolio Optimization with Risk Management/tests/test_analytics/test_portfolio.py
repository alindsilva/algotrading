"""
Unit tests for the portfolio analytics module.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch

from src.analytics.portfolio import PortfolioAnalytics, PortfolioMetrics
from src.core.exceptions import ValidationError
from tests.conftest import assert_almost_equal


class TestPortfolioAnalytics:
    """Test PortfolioAnalytics class initialization and basic functionality."""
    
    def test_init_default(self):
        """Test default initialization."""
        analytics = PortfolioAnalytics()
        
        assert analytics.risk_free_rate == 0.0
        assert analytics.benchmark_returns is None
        assert analytics.window_size == 252
        assert_almost_equal(analytics._daily_risk_free_rate, 0.0, tolerance=1e-8)
    
    def test_init_custom(self):
        """Test initialization with custom parameters."""
        benchmark = pd.Series([0.001, 0.002, -0.001])
        analytics = PortfolioAnalytics(
            risk_free_rate=0.03,
            benchmark_returns=benchmark,
            window_size=21
        )
        
        assert analytics.risk_free_rate == 0.03
        assert analytics.benchmark_returns is benchmark
        assert analytics.window_size == 21
        
        # Test daily risk-free rate calculation
        expected_daily_rate = (1 + 0.03) ** (1/252) - 1
        assert_almost_equal(analytics._daily_risk_free_rate, expected_daily_rate)


class TestPortfolioMetricsCalculation:
    """Test portfolio metrics calculation methods."""
    
    @pytest.mark.asyncio
    async def test_calculate_returns_from_prices(self, portfolio_analytics, sample_prices):
        """Test calculation of returns from price series."""
        returns = await portfolio_analytics._calculate_returns_from_prices(sample_prices)
        
        # Check that returns series is correct
        assert isinstance(returns, pd.Series)
        assert len(returns) == len(sample_prices) - 1  # One less than prices
        
        # Check that returns calculation is correct
        expected_first_return = (sample_prices.iloc[1] / sample_prices.iloc[0]) - 1
        assert_almost_equal(returns.iloc[0], expected_first_return)
    
    @pytest.mark.asyncio
    async def test_calculate_returns_from_prices_invalid_input(self, portfolio_analytics):
        """Test returns calculation with invalid input."""
        # Test with empty series
        with pytest.raises(ValidationError, match="Insufficient price data"):
            await portfolio_analytics._calculate_returns_from_prices(pd.Series([]))
        
        # Test with single price
        with pytest.raises(ValidationError, match="Insufficient price data"):
            await portfolio_analytics._calculate_returns_from_prices(pd.Series([100.0]))
        
        # Test with non-pandas series
        with pytest.raises(ValidationError, match="Prices must be a pandas Series"):
            await portfolio_analytics._calculate_returns_from_prices([100, 101, 102])
    
    @pytest.mark.asyncio
    async def test_calculate_performance_metrics(self, portfolio_analytics, sample_returns):
        """Test performance metrics calculation."""
        metrics = await portfolio_analytics._calculate_performance_metrics(sample_returns)
        
        # Check that all metrics are present
        assert 'total_return' in metrics
        assert 'annualized_return' in metrics
        assert 'cumulative_returns' in metrics
        
        # Check cumulative returns
        assert isinstance(metrics['cumulative_returns'], pd.Series)
        assert len(metrics['cumulative_returns']) == len(sample_returns)
        
        # Total return should be cumulative return - 1
        expected_total_return = metrics['cumulative_returns'].iloc[-1] - 1
        assert_almost_equal(metrics['total_return'], expected_total_return)
        
        # Annualized return should be reasonable
        assert isinstance(metrics['annualized_return'], float)
        assert -1 < metrics['annualized_return'] < 5  # Reasonable range
    
    @pytest.mark.asyncio
    async def test_calculate_risk_metrics(self, portfolio_analytics, sample_returns):
        """Test risk metrics calculation."""
        metrics = await portfolio_analytics._calculate_risk_metrics(sample_returns)
        
        # Check that all metrics are present
        expected_keys = [
            'volatility', 'max_drawdown', 'sharpe_ratio', 
            'sortino_ratio', 'calmar_ratio', 'omega_ratio'
        ]
        for key in expected_keys:
            assert key in metrics
        
        # Check volatility (should be positive)
        assert metrics['volatility'] > 0
        assert metrics['volatility'] < 2  # Reasonable upper bound
        
        # Check max drawdown (should be between 0 and 1)
        assert 0 <= metrics['max_drawdown'] <= 1
        
        # Check Sharpe ratio (should be finite)
        assert np.isfinite(metrics['sharpe_ratio'])
        
        # Check other ratios
        assert np.isfinite(metrics['sortino_ratio'])
        assert metrics['omega_ratio'] >= 0  # Omega ratio should be non-negative
    
    @pytest.mark.asyncio
    async def test_calculate_tail_risk_metrics(self, portfolio_analytics, sample_returns):
        """Test tail risk metrics calculation."""
        metrics = await portfolio_analytics._calculate_tail_risk_metrics(sample_returns)
        
        # Check that metrics are present
        assert 'var_95' in metrics
        assert 'cvar_95' in metrics
        
        # VaR should be positive (we take absolute value)
        assert metrics['var_95'] >= 0
        assert metrics['cvar_95'] >= 0
        
        # CVaR should generally be greater than or equal to VaR
        # (in terms of absolute loss)
        assert metrics['cvar_95'] >= metrics['var_95']
    
    @pytest.mark.asyncio
    async def test_calculate_correlation_metrics(self, sample_returns):
        """Test correlation metrics calculation."""
        # Create benchmark returns
        np.random.seed(123)
        benchmark_returns = pd.Series(
            np.random.normal(0.0005, 0.015, len(sample_returns)),
            index=sample_returns.index
        )
        
        analytics = PortfolioAnalytics(benchmark_returns=benchmark_returns)
        metrics = await analytics._calculate_correlation_metrics(sample_returns)
        
        # Check that metrics are present
        assert 'beta' in metrics
        assert 'correlation' in metrics
        
        # Check that values are reasonable
        assert np.isfinite(metrics['beta'])
        assert np.isfinite(metrics['correlation'])
        assert -1 <= metrics['correlation'] <= 1
    
    @pytest.mark.asyncio
    async def test_calculate_correlation_metrics_no_benchmark(self, portfolio_analytics, sample_returns):
        """Test correlation metrics with no benchmark."""
        metrics = await portfolio_analytics._calculate_correlation_metrics(sample_returns)
        
        assert metrics['beta'] is None
        assert metrics['correlation'] is None


class TestFullPortfolioMetrics:
    """Test full portfolio metrics calculation."""
    
    @pytest.mark.asyncio
    async def test_calculate_portfolio_metrics_from_returns(self, portfolio_analytics, sample_returns):
        """Test full portfolio metrics calculation from returns."""
        metrics = await portfolio_analytics.calculate_portfolio_metrics(sample_returns)
        
        # Check that it's a PortfolioMetrics object
        assert isinstance(metrics, PortfolioMetrics)
        
        # Check all required fields are present
        assert hasattr(metrics, 'total_return')
        assert hasattr(metrics, 'annualized_return')
        assert hasattr(metrics, 'volatility')
        assert hasattr(metrics, 'max_drawdown')
        assert hasattr(metrics, 'sharpe_ratio')
        assert hasattr(metrics, 'sortino_ratio')
        assert hasattr(metrics, 'calmar_ratio')
        assert hasattr(metrics, 'omega_ratio')
        assert hasattr(metrics, 'var_95')
        assert hasattr(metrics, 'cvar_95')
        
        # Check that values are reasonable
        assert np.isfinite(metrics.total_return)
        assert np.isfinite(metrics.annualized_return)
        assert metrics.volatility > 0
        assert 0 <= metrics.max_drawdown <= 1
        
        # Check timestamp is set
        assert isinstance(metrics.calculation_time, datetime)
        assert metrics.time_period_days == len(sample_returns)
    
    @pytest.mark.asyncio
    async def test_calculate_portfolio_metrics_from_prices(self, portfolio_analytics, sample_prices):
        """Test portfolio metrics calculation from prices."""
        metrics = await portfolio_analytics.calculate_portfolio_metrics(
            returns=None, 
            prices=sample_prices
        )
        
        assert isinstance(metrics, PortfolioMetrics)
        assert metrics.time_period_days == len(sample_prices) - 1  # Returns are one less than prices
    
    @pytest.mark.asyncio
    async def test_calculate_portfolio_metrics_numpy_array(self, portfolio_analytics, sample_returns):
        """Test portfolio metrics calculation with numpy array input."""
        returns_array = sample_returns.values
        
        metrics = await portfolio_analytics.calculate_portfolio_metrics(returns_array)
        
        assert isinstance(metrics, PortfolioMetrics)
        assert metrics.time_period_days == len(returns_array)
    
    @pytest.mark.asyncio
    async def test_calculate_portfolio_metrics_insufficient_data(self, portfolio_analytics):
        """Test portfolio metrics calculation with insufficient data."""
        # Test with very few data points
        short_returns = pd.Series([0.01, 0.02, -0.01])
        
        with pytest.raises(ValidationError, match="Insufficient data"):
            await portfolio_analytics.calculate_portfolio_metrics(short_returns)
    
    @pytest.mark.asyncio
    async def test_calculate_portfolio_metrics_invalid_input(self, portfolio_analytics):
        """Test portfolio metrics calculation with invalid input."""
        with pytest.raises(ValidationError, match="Returns must be a pandas Series or numpy array"):
            await portfolio_analytics.calculate_portfolio_metrics("invalid_input")


class TestPositionAnalysis:
    """Test position analysis functionality."""
    
    @pytest.mark.asyncio
    async def test_analyze_positions_basic(self, portfolio_analytics, sample_positions, sample_account_values):
        """Test basic position analysis."""
        analysis = await portfolio_analytics.analyze_positions(sample_positions, sample_account_values)
        
        # Check that all expected fields are present
        expected_fields = [
            'total_positions', 'total_value', 'long_exposure', 'short_exposure',
            'net_exposure', 'gross_exposure', 'positions_by_type', 'largest_positions'
        ]
        for field in expected_fields:
            assert field in analysis
        
        # Check position count
        assert analysis['total_positions'] == len(sample_positions)
        
        # Check exposure calculations
        assert analysis['long_exposure'] >= 0
        assert analysis['short_exposure'] <= 0  # Should be negative or zero
        assert isinstance(analysis['gross_exposure'], float)
        
        # Check largest positions
        assert isinstance(analysis['largest_positions'], list)
        assert len(analysis['largest_positions']) <= 10  # Should be top 10 or fewer
    
    @pytest.mark.asyncio
    async def test_analyze_positions_empty(self, portfolio_analytics, sample_account_values):
        """Test position analysis with empty positions."""
        analysis = await portfolio_analytics.analyze_positions([], sample_account_values)
        
        assert analysis['total_positions'] == 0
        assert analysis['total_value'] == 0.0
        assert analysis['long_exposure'] == 0.0
        assert analysis['short_exposure'] == 0.0
        assert analysis['net_exposure'] == 0.0
        assert analysis['gross_exposure'] == 0.0
        assert analysis['positions_by_type'] == {}
        assert analysis['largest_positions'] == []
    
    @pytest.mark.asyncio
    async def test_analyze_positions_calculations(self, portfolio_analytics, sample_positions, sample_account_values):
        """Test position analysis calculations."""
        analysis = await portfolio_analytics.analyze_positions(sample_positions, sample_account_values)
        
        # Manual calculation verification
        total_long = 0
        total_short = 0
        
        for pos in sample_positions:
            position_value = pos.position * pos.avg_cost
            if pos.position > 0:
                total_long += position_value
            else:
                total_short += position_value  # This will be negative
        
        assert_almost_equal(analysis['long_exposure'], total_long)
        assert_almost_equal(analysis['short_exposure'], total_short)
        assert_almost_equal(
            analysis['net_exposure'], 
            total_long + total_short  # Short is already negative
        )
        assert_almost_equal(
            analysis['gross_exposure'], 
            total_long + abs(total_short)
        )


class TestRollingMetrics:
    """Test rolling metrics calculation."""
    
    @pytest.mark.asyncio
    async def test_calculate_rolling_metrics_basic(self, portfolio_analytics, sample_returns):
        """Test basic rolling metrics calculation."""
        # Use a small window for testing
        window = 21
        rolling_metrics = await portfolio_analytics.calculate_rolling_metrics(
            sample_returns, window=window
        )
        
        # Check that all expected metrics are present
        expected_metrics = [
            'rolling_volatility', 'rolling_sharpe_ratio', 
            'rolling_sortino_ratio', 'rolling_max_drawdown'
        ]
        for metric in expected_metrics:
            assert metric in rolling_metrics
            assert isinstance(rolling_metrics[metric], pd.Series)
        
        # Check that series have correct length
        expected_length = len(sample_returns) - window + 1
        assert len(rolling_metrics['rolling_volatility']) == expected_length
    
    @pytest.mark.asyncio
    async def test_calculate_rolling_metrics_insufficient_data(self, portfolio_analytics):
        """Test rolling metrics with insufficient data."""
        short_returns = pd.Series([0.01, 0.02, -0.01, 0.005])
        window = 10
        
        with pytest.raises(ValidationError, match="Insufficient return data"):
            await portfolio_analytics.calculate_rolling_metrics(short_returns, window=window)
    
    @pytest.mark.asyncio
    async def test_calculate_rolling_metrics_default_window(self, sample_returns):
        """Test rolling metrics with default window size."""
        analytics = PortfolioAnalytics(window_size=50)  # Smaller than default for testing
        
        rolling_metrics = await analytics.calculate_rolling_metrics(sample_returns)
        
        # Should use the default window size from analytics instance
        expected_length = len(sample_returns) - 50 + 1
        assert len(rolling_metrics['rolling_volatility']) == expected_length


class TestStressTesting:
    """Test stress testing functionality."""
    
    @pytest.mark.asyncio
    async def test_stress_test_portfolio_default_scenarios(self, portfolio_analytics, sample_returns):
        """Test stress testing with default scenarios."""
        stress_results = await portfolio_analytics.stress_test_portfolio(sample_returns)
        
        # Check that results contain expected scenarios
        expected_scenarios = [
            'market_crash_15pct', 'market_crash_10pct', 'market_crash_5pct',
            'interest_rate_increase', 'volatility_spike'
        ]
        for scenario in expected_scenarios:
            assert scenario in stress_results
        
        # Check that each scenario has expected fields
        for scenario, results in stress_results.items():
            expected_fields = [
                'new_sharpe_ratio', 'sharpe_ratio_change',
                'new_max_drawdown', 'max_drawdown_change',
                'new_volatility', 'volatility_change'
            ]
            for field in expected_fields:
                assert field in results
                assert np.isfinite(results[field])
    
    @pytest.mark.asyncio
    async def test_stress_test_portfolio_custom_scenarios(self, portfolio_analytics, sample_returns):
        """Test stress testing with custom scenarios."""
        custom_scenarios = {
            'mild_shock': -0.02,
            'severe_shock': -0.20
        }
        
        stress_results = await portfolio_analytics.stress_test_portfolio(
            sample_returns, scenarios=custom_scenarios
        )
        
        # Check that custom scenarios are in results
        assert 'mild_shock' in stress_results
        assert 'severe_shock' in stress_results
        
        # Severe shock should generally have worse impact than mild shock
        mild_sharpe_change = stress_results['mild_shock']['sharpe_ratio_change']
        severe_sharpe_change = stress_results['severe_shock']['sharpe_ratio_change']
        
        # More severe shock should generally have more negative impact on Sharpe ratio
        assert severe_sharpe_change <= mild_sharpe_change


class TestOmegaRatio:
    """Test Omega ratio calculation."""
    
    def test_omega_ratio_positive_returns(self, portfolio_analytics):
        """Test Omega ratio with all positive returns."""
        positive_returns = pd.Series([0.01, 0.02, 0.015, 0.005, 0.012])
        threshold = 0.0
        
        omega = portfolio_analytics._calculate_omega_ratio(positive_returns, threshold)
        
        # With all positive returns and zero threshold, omega should be infinite
        assert omega == float('inf')
    
    def test_omega_ratio_mixed_returns(self, portfolio_analytics):
        """Test Omega ratio with mixed returns."""
        mixed_returns = pd.Series([0.02, -0.01, 0.015, -0.005, 0.01])
        threshold = 0.0
        
        omega = portfolio_analytics._calculate_omega_ratio(mixed_returns, threshold)
        
        # Should be a finite positive number
        assert np.isfinite(omega)
        assert omega > 0
        
        # Calculate manually to verify
        positive_sum = mixed_returns[mixed_returns > threshold].sum()
        negative_sum = abs(mixed_returns[mixed_returns < threshold].sum())
        expected_omega = positive_sum / negative_sum
        
        assert_almost_equal(omega, expected_omega)
    
    def test_omega_ratio_all_negative_returns(self, portfolio_analytics):
        """Test Omega ratio with all negative returns."""
        negative_returns = pd.Series([-0.01, -0.02, -0.015, -0.005])
        threshold = 0.0
        
        omega = portfolio_analytics._calculate_omega_ratio(negative_returns, threshold)
        
        # With all negative returns, omega should be 0
        assert omega == 0.0


class TestPortfolioAnalyticsIntegration:
    """Integration tests for portfolio analytics."""
    
    @pytest.mark.asyncio
    async def test_full_analytics_workflow(self, sample_returns, sample_positions, sample_account_values):
        """Test complete analytics workflow."""
        # Create benchmark returns
        np.random.seed(42)
        benchmark_returns = pd.Series(
            np.random.normal(0.0005, 0.015, len(sample_returns)),
            index=sample_returns.index
        )
        
        # Initialize analytics with benchmark
        analytics = PortfolioAnalytics(
            risk_free_rate=0.025,
            benchmark_returns=benchmark_returns,
            window_size=21
        )
        
        # Calculate portfolio metrics
        metrics = await analytics.calculate_portfolio_metrics(sample_returns)
        
        # Analyze positions
        position_analysis = await analytics.analyze_positions(
            sample_positions, sample_account_values
        )
        
        # Calculate rolling metrics
        rolling_metrics = await analytics.calculate_rolling_metrics(sample_returns)
        
        # Perform stress testing
        stress_results = await analytics.stress_test_portfolio(sample_returns)
        
        # Verify all components work together
        assert isinstance(metrics, PortfolioMetrics)
        assert metrics.beta_to_market is not None  # Should have benchmark metrics
        assert metrics.correlation_to_market is not None
        
        assert position_analysis['total_positions'] > 0
        assert len(rolling_metrics) > 0
        assert len(stress_results) > 0
    
    @pytest.mark.asyncio  
    async def test_analytics_with_extreme_data(self, portfolio_analytics):
        """Test analytics with extreme market conditions."""
        # Create extreme returns (market crash scenario)
        extreme_returns = pd.Series([
            -0.10, -0.15, -0.08, -0.12, -0.20,  # Crash period
            0.05, 0.08, 0.03, 0.07, 0.04,       # Recovery
            0.001, 0.002, -0.001, 0.003, 0.0    # Normal period
        ])
        
        metrics = await portfolio_analytics.calculate_portfolio_metrics(extreme_returns)
        
        # Verify metrics handle extreme values appropriately
        assert np.isfinite(metrics.total_return)
        assert np.isfinite(metrics.volatility)
        assert metrics.volatility > 0
        assert 0 <= metrics.max_drawdown <= 1
        
        # Max drawdown should be significant given the crash
        assert metrics.max_drawdown > 0.1  # At least 10% drawdown
    
    @pytest.mark.asyncio
    async def test_analytics_performance(self, benchmark_timer):
        """Test analytics performance with large dataset."""
        # Create large dataset
        np.random.seed(42)
        large_returns = pd.Series(np.random.normal(0.001, 0.02, 5000))
        
        analytics = PortfolioAnalytics()
        
        # Benchmark the calculation
        benchmark_timer.start()
        metrics = await analytics.calculate_portfolio_metrics(large_returns)
        elapsed = benchmark_timer.stop()
        
        # Should complete within reasonable time (less than 1 second)
        assert elapsed < 1.0
        assert isinstance(metrics, PortfolioMetrics)
