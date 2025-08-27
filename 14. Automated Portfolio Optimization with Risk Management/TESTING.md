# Testing Guide

This document provides comprehensive information about testing the IBKR Trading Application.

## Overview

The testing framework is built using pytest with comprehensive coverage for:
- Unit tests for individual components
- Integration tests for system workflows  
- Async testing support
- Coverage reporting
- Code quality checks

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and utilities
├── test_core/              # Core module tests
│   └── test_config.py      # Configuration tests
├── test_contracts/         # Contract factory tests  
│   └── test_factory.py     # Contract creation tests
├── test_analytics/         # Portfolio analytics tests
│   └── test_portfolio.py   # Risk metrics tests
├── test_data/              # Data storage tests
│   └── test_storage.py     # Database operations tests
└── test_app/               # Application integration tests
    └── test_main.py        # Main application tests
```

## Running Tests

### Quick Start

```bash
# Run all tests with coverage
python run_tests.py test

# Run specific test types
python run_tests.py test --test-type unit
python run_tests.py test --test-type integration

# Run without coverage
python run_tests.py test --no-coverage

# Verbose output
python run_tests.py test --verbose
```

### Advanced Test Running

```bash
# Run complete test suite (format, lint, type-check, test)
python run_tests.py all

# Run specific actions
python run_tests.py lint
python run_tests.py type-check
python run_tests.py format

# Clean cache files
python run_tests.py clean
```

### Direct pytest Usage

```bash
# Basic test run
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing --cov-report=html

# Run specific test file
pytest tests/test_core/test_config.py

# Run tests matching pattern
pytest -k "test_portfolio" -v

# Run only unit tests
pytest -m unit

# Stop on first failure
pytest -x tests/
```

## Test Categories

### Unit Tests

Test individual components in isolation:

```bash
# Run all unit tests
pytest tests/test_core tests/test_contracts tests/test_analytics tests/test_data -m unit

# Specific module unit tests
pytest tests/test_core/ -v
pytest tests/test_contracts/ -v
pytest tests/test_analytics/ -v
```

**Coverage areas:**
- Configuration management
- Contract factory functionality
- Portfolio analytics calculations
- Database operations
- Type system and exceptions

### Integration Tests

Test system workflows and component interactions:

```bash
# Run integration tests
pytest tests/test_app/ -m integration

# Application lifecycle tests
pytest tests/test_app/test_main.py::TestIBKRAppLifecycle -v
```

**Coverage areas:**
- Application startup/shutdown
- Full portfolio summary generation
- Market data streaming workflows
- Risk report generation
- Background task management

### Async Tests

All async functionality is tested using pytest-asyncio:

```python
@pytest.mark.asyncio
async def test_async_function():
    result = await some_async_function()
    assert result is not None
```

## Test Fixtures

### Core Fixtures

- `temp_dir`: Temporary directory for test files
- `test_config`: Test configuration with safe defaults
- `test_storage`: Initialized test database
- `mock_ibkr_client`: Mocked IBKR client for testing

### Data Fixtures

- `sample_positions`: Mock position data
- `sample_account_values`: Mock account values  
- `sample_returns`: Realistic return series for analytics
- `sample_prices`: Price series derived from returns
- `sample_contracts`: Various contract types for testing

### Analytics Fixtures

- `portfolio_analytics`: Portfolio analytics instance
- `benchmark_timer`: Performance timing utility

## Mocking Strategy

### IBKR API Mocking

The IBKR API is mocked at multiple levels:

```python
# Mock entire ibapi package
@pytest.fixture
def mock_ibapi():
    with patch('src.api.client.EClient'), \
         patch('src.api.client.EWrapper'):
        yield

# Mock client responses
mock_client = AsyncMock()
mock_client.get_positions.return_value = sample_positions
mock_client.get_account_values.return_value = sample_account_values
```

### Database Mocking

Tests use temporary SQLite databases:

```python
@pytest.fixture
async def test_storage(test_config):
    storage = AsyncDataStorage(test_config.database)
    await storage.initialize()
    yield storage
    await storage.close()
```

### Time Mocking

For time-sensitive tests, we can use freezegun:

```python
from freezegun import freeze_time

@freeze_time("2023-01-15 10:30:00")
def test_time_sensitive_function():
    # Test with fixed time
    pass
```

## Coverage Requirements

The test suite maintains **minimum 80% code coverage**:

```bash
# Check coverage
pytest tests/ --cov=src --cov-report=term-missing --cov-fail-under=80

# Generate HTML coverage report
pytest tests/ --cov=src --cov-report=html
# View: open htmlcov/index.html
```

### Coverage Exclusions

Certain patterns are excluded from coverage:

```python
# pragma: no cover - exclude from coverage
def debug_function():  # pragma: no cover
    print("Debug info")
```

## Performance Testing

### Benchmarking

Use the benchmark timer fixture for performance tests:

```python
@pytest.mark.asyncio
async def test_analytics_performance(benchmark_timer):
    benchmark_timer.start()
    result = await expensive_calculation()
    elapsed = benchmark_timer.stop()
    
    assert elapsed < 1.0  # Should complete in less than 1 second
```

### Load Testing

For testing concurrent operations:

```python
@pytest.mark.asyncio
async def test_concurrent_database_operations():
    tasks = [store_data(f"symbol_{i}") for i in range(100)]
    results = await asyncio.gather(*tasks)
    assert len(results) == 100
```

## Test Data Management

### Realistic Test Data

Tests use realistic financial data:

```python
# Generate realistic returns
np.random.seed(42)  # Reproducible
returns = np.random.normal(0.0008, 0.02, 252)  # ~20% annual return, 20% volatility

# Add market patterns
returns[50:60] = -0.05  # Market crash
returns[200:210] = 0.03  # Bull run
```

### Data Factories

For complex object creation:

```python
def create_position_data(symbol="AAPL", position=100.0, avg_cost=150.0):
    return PositionData(
        account="DU123456",
        symbol=symbol,
        sec_type="STK",
        exchange="NASDAQ", 
        currency="USD",
        position=position,
        avg_cost=avg_cost,
        timestamp=datetime.now()
    )
```

## Debugging Tests

### Verbose Output

```bash
# Detailed test output
pytest tests/ -v -s

# Show print statements
pytest tests/ -s

# Show local variables on failure
pytest tests/ -l
```

### Interactive Debugging

```bash
# Drop into debugger on failure
pytest tests/ --pdb

# Drop into debugger on first failure
pytest tests/ -x --pdb
```

### Logging in Tests

```python
import logging

def test_with_logging(caplog):
    with caplog.at_level(logging.INFO):
        function_that_logs()
    
    assert "Expected log message" in caplog.text
```

## Continuous Integration

### GitHub Actions

Tests run automatically on:
- Pull requests
- Pushes to main branch
- Scheduled runs (daily)

### Test Matrix

Tests run across:
- Python 3.8, 3.9, 3.10, 3.11
- Different dependency versions
- Various operating systems

## Common Testing Patterns

### Testing Async Code

```python
@pytest.mark.asyncio
async def test_async_operation():
    result = await async_function()
    assert result is not None
```

### Testing Exceptions

```python
def test_exception_handling():
    with pytest.raises(ValidationError, match="Expected error message"):
        function_that_should_raise()
```

### Parametrized Tests

```python
@pytest.mark.parametrize("input,expected", [
    ("AAPL", "AAPL"),
    ("aapl", "AAPL"),
    ("Apple", ValidationError),
])
def test_symbol_normalization(input, expected):
    if expected == ValidationError:
        with pytest.raises(ValidationError):
            normalize_symbol(input)
    else:
        assert normalize_symbol(input) == expected
```

### Testing Database Operations

```python
@pytest.mark.asyncio
async def test_database_operation(test_storage):
    # Insert test data
    await test_storage.store_market_data(
        symbol="AAPL",
        timestamp=datetime.now(),
        price=150.0,
        size=100,
        tick_type=1
    )
    
    # Verify data
    data = await test_storage.get_market_data("AAPL")
    assert len(data) == 1
    assert data[0][1] == "AAPL"  # symbol
```

## Test Maintenance

### Updating Tests

When adding new features:

1. Write tests first (TDD)
2. Ensure new code has >80% coverage
3. Update fixtures if needed
4. Add integration tests for workflows

### Cleaning Up Tests

```bash
# Remove obsolete test files
# Update fixture data
# Refactor common patterns
```

### Performance Monitoring

Monitor test execution time:

```bash
# Show slowest tests
pytest tests/ --durations=10

# Profile test execution
pytest tests/ --profile
```

## Troubleshooting

### Common Issues

1. **Async test failures**: Ensure `@pytest.mark.asyncio` decorator
2. **Database locks**: Use separate test databases
3. **Timing issues**: Use fixed time with freezegun
4. **Import errors**: Check Python path and module structure

### Test Environment

Ensure clean test environment:

```bash
# Clean cache
python run_tests.py clean

# Reinstall dependencies
pip install -r requirements.txt

# Run in isolated environment
python -m venv test_env
source test_env/bin/activate
pip install -r requirements.txt
```

## Best Practices

1. **Test Independence**: Each test should be independent
2. **Descriptive Names**: Use clear, descriptive test names
3. **Single Responsibility**: One assertion per test when possible
4. **Fast Tests**: Keep unit tests fast (<100ms)
5. **Realistic Data**: Use realistic test data
6. **Mock External Dependencies**: Mock IBKR API, file system, network
7. **Test Edge Cases**: Include boundary conditions and error cases
8. **Documentation**: Document complex test scenarios

## Examples

### Complete Test Example

```python
import pytest
from unittest.mock import AsyncMock, patch
from src.analytics.portfolio import PortfolioAnalytics

class TestPortfolioAnalytics:
    """Test portfolio analytics functionality."""
    
    @pytest.mark.asyncio
    async def test_calculate_sharpe_ratio(self, sample_returns):
        """Test Sharpe ratio calculation with known data."""
        analytics = PortfolioAnalytics(risk_free_rate=0.02)
        
        metrics = await analytics.calculate_portfolio_metrics(sample_returns)
        
        # Verify reasonable Sharpe ratio
        assert -2 < metrics.sharpe_ratio < 5
        assert isinstance(metrics.sharpe_ratio, float)
    
    def test_invalid_input_handling(self):
        """Test handling of invalid inputs."""
        analytics = PortfolioAnalytics()
        
        with pytest.raises(ValidationError):
            analytics._calculate_omega_ratio("invalid", 0.0)
```

This testing framework provides comprehensive coverage while maintaining fast execution and clear structure for ongoing development and maintenance.
