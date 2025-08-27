# Test Case Commands Reference

This document provides comprehensive instructions for running test cases in the IBKR Automated Portfolio Optimization project.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Basic Test Commands](#basic-test-commands)
- [Module-Specific Tests](#module-specific-tests)
- [Class and Method-Specific Tests](#class-and-method-specific-tests)
- [Advanced Test Options](#advanced-test-options)
- [Coverage Reports](#coverage-reports)
- [Test Filtering](#test-filtering)
- [Debugging Options](#debugging-options)
- [Performance Testing](#performance-testing)
- [Development Workflows](#development-workflows)
- [Common Test Patterns](#common-test-patterns)
- [Troubleshooting](#troubleshooting)

## Prerequisites

Ensure you're in the project root directory and have the necessary dependencies:

```bash
# Navigate to project root
cd "/Users/alin/Documents/github/algotrading/14. Automated Portfolio Optimization with Risk Management"

# Activate conda environment
conda activate my_quant_stack

# Install test dependencies
pip install pytest pytest-asyncio pytest-cov pytest-xdist pytest-mock
```

## Basic Test Commands

### Run All Tests
```bash
# Run all tests with verbose output
python -m pytest tests/ -v

# Run all tests with clean short traceback
python -m pytest tests/ -v --tb=short

# Run all tests with minimal output
python -m pytest tests/ -q

# Run all tests and stop on first failure
python -m pytest tests/ -x
```

## Module-Specific Tests

### Core Application Tests
```bash
# Main IBKR application tests
python -m pytest tests/test_app/test_main.py -v

# Portfolio analytics tests
python -m pytest tests/test_analytics/test_portfolio.py -v

# Data storage tests  
python -m pytest tests/test_data/test_storage.py -v

# Core configuration tests
python -m pytest tests/test_core/test_config.py -v

# Contract factory tests
python -m pytest tests/test_contracts/test_factory.py -v
```

### New Trading Functionality Tests
```bash
# Order management tests
python -m pytest tests/test_orders/test_manager.py -v

# API client tests
python -m pytest tests/test_api/test_client.py -v
```

## Class and Method-Specific Tests

### Trading Methods Tests
```bash
# All trading methods
python -m pytest tests/test_app/test_main.py::TestTradingMethods -v

# Portfolio management methods
python -m pytest tests/test_app/test_main.py::TestPortfolioMethods -v

# Market data methods
python -m pytest tests/test_app/test_main.py::TestMarketDataMethods -v

# Error handling tests
python -m pytest tests/test_app/test_main.py::TestErrorHandling -v
```

### Specific Trading Functions
```bash
# Target percentage orders
python -m pytest tests/test_app/test_main.py::TestTradingMethods::test_order_target_percent -v

# Target quantity orders
python -m pytest tests/test_app/test_main.py::TestTradingMethods::test_order_target_quantity -v

# Target value orders
python -m pytest tests/test_app/test_main.py::TestTradingMethods::test_order_target_value -v

# Portfolio rebalancing
python -m pytest tests/test_app/test_main.py::TestPortfolioMethods::test_rebalance_portfolio -v

# Portfolio allocations
python -m pytest tests/test_app/test_main.py::TestPortfolioMethods::test_get_portfolio_allocations -v
```

### Order Manager Tests
```bash
# Basic order placement
python -m pytest tests/test_orders/test_manager.py::TestBasicOrderPlacement -v

# Advanced order methods
python -m pytest tests/test_orders/test_manager.py::TestAdvancedOrderMethods -v

# Portfolio rebalancing
python -m pytest tests/test_orders/test_manager.py::TestPortfolioRebalancing -v

# Order validation
python -m pytest tests/test_orders/test_manager.py::TestOrderValidation -v

# Order cancellation
python -m pytest tests/test_orders/test_manager.py::TestOrderCancellation -v
```

## Advanced Test Options

### Coverage Reports
```bash
# Generate HTML coverage report
python -m pytest tests/ --cov=src --cov-report=html

# Show coverage with missing lines
python -m pytest tests/ --cov=src --cov-report=term-missing

# Coverage for specific module
python -m pytest tests/test_orders/ --cov=src.orders --cov-report=term-missing

# Minimum coverage threshold
python -m pytest tests/ --cov=src --cov-fail-under=80
```

### Test Output Formatting
```bash
# Extra verbose output
python -m pytest tests/ -vv

# Show full traceback
python -m pytest tests/ --tb=long

# Show only short traceback
python -m pytest tests/ --tb=short

# Show no traceback
python -m pytest tests/ --tb=no

# Show local variables in traceback
python -m pytest tests/ -l --tb=short
```

## Test Filtering

### Keyword Filtering
```bash
# Run tests containing "order"
python -m pytest tests/ -k "order" -v

# Run tests containing "portfolio" 
python -m pytest tests/ -k "portfolio" -v

# Run tests containing "target"
python -m pytest tests/ -k "target" -v

# Run async tests only
python -m pytest tests/ -k "async" -v

# Exclude slow tests
python -m pytest tests/ -k "not slow" -v

# Multiple keywords (OR)
python -m pytest tests/ -k "order or portfolio" -v

# Multiple keywords (AND)
python -m pytest tests/ -k "order and target" -v
```

### Marker-Based Filtering
```bash
# Run only integration tests (if marked)
python -m pytest tests/ -m integration -v

# Run only unit tests (if marked)
python -m pytest tests/ -m unit -v

# Skip slow tests (if marked)
python -m pytest tests/ -m "not slow" -v
```

## Debugging Options

### Interactive Debugging
```bash
# Drop into debugger on failure
python -m pytest tests/ --pdb

# Drop into debugger on first failure
python -m pytest tests/ --pdb -x

# Show print statements
python -m pytest tests/ -s

# Capture method (show print for failed tests only)
python -m pytest tests/ --capture=no
```

### Failure Analysis
```bash
# Stop after N failures
python -m pytest tests/ --maxfail=3

# Re-run only failed tests from last run
python -m pytest tests/ --lf

# Re-run failed tests first, then continue
python -m pytest tests/ --ff

# Show slowest N test durations
python -m pytest tests/ --durations=10
```

## Performance Testing

### Parallel Execution
```bash
# Run tests in parallel (auto-detect CPU cores)
python -m pytest tests/ -n auto

# Run tests with specific number of workers
python -m pytest tests/ -n 4

# Distribute tests by file
python -m pytest tests/ -n auto --dist loadfile
```

### Performance Monitoring
```bash
# Show test durations
python -m pytest tests/ --durations=0

# Show only slowest tests
python -m pytest tests/ --durations=5

# Profile test execution
python -m pytest tests/ --profile
```

## Development Workflows

### Quick Development Testing
```bash
# Test only new trading functionality
python -m pytest tests/test_app/test_main.py::TestTradingMethods tests/test_app/test_main.py::TestPortfolioMethods -v --tb=short

# Fast iteration during development
python -m pytest tests/test_orders/test_manager.py::TestBasicOrderPlacement -v -x

# Test specific feature during development
python -m pytest tests/ -k "target_percent" -v --tb=short
```

### Pre-Commit Testing
```bash
# Run full test suite with coverage
python -m pytest tests/ --cov=src --cov-report=term-missing -v

# Run tests with strict warnings
python -m pytest tests/ -W error::DeprecationWarning

# Run tests with quality checks
python -m pytest tests/ --cov=src --cov-fail-under=80 --maxfail=5
```

### Integration Testing
```bash
# Run integration tests only
python -m pytest tests/test_app/test_main.py::TestApplicationIntegration -v

# Run workflow tests
python -m pytest tests/ -k "workflow or integration" -v

# Run end-to-end tests
python -m pytest tests/test_app/test_main.py::TestApplicationIntegration::test_trading_workflow_integration -v
```

## Common Test Patterns

### Testing New Features
```bash
# Test all new trading methods
python -m pytest tests/ -k "order_target or order_percent or order_value or rebalance" -v

# Test portfolio management features
python -m pytest tests/ -k "portfolio or allocation or rebalance" -v

# Test order management features  
python -m pytest tests/ -k "cancel or update or place_order" -v
```

### Error and Edge Case Testing
```bash
# Test error handling
python -m pytest tests/ -k "error" -v

# Test connection issues
python -m pytest tests/ -k "connection" -v

# Test validation
python -m pytest tests/ -k "validation" -v
```

### Component Testing
```bash
# Test core components
python -m pytest tests/test_core/ tests/test_contracts/ tests/test_data/ -v

# Test new components
python -m pytest tests/test_orders/ tests/test_api/ -v

# Test analytics and application
python -m pytest tests/test_analytics/ tests/test_app/ -v
```

## Test Configuration Files

### pytest.ini Configuration
The project uses `pytest.ini` for default settings:

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
asyncio_mode = auto
addopts = 
    --strict-markers
    --strict-config
    --verbose
filterwarnings =
    ignore::DeprecationWarning
    ignore::PendingDeprecationWarning
```

### Custom Configuration Override
```bash
# Override configuration temporarily
python -m pytest tests/ --asyncio-mode=strict -v --tb=short --maxfail=5

# Use custom configuration file
python -m pytest tests/ -c custom_pytest.ini
```

## Troubleshooting

### Common Issues and Solutions

#### Async Test Issues
```bash
# If async tests fail, ensure proper mode
python -m pytest tests/ --asyncio-mode=auto

# For strict async testing
python -m pytest tests/ --asyncio-mode=strict
```

#### Import Issues
```bash
# Run tests with Python path
PYTHONPATH=. python -m pytest tests/ -v

# Install package in development mode
pip install -e .
```

#### Mock Issues
```bash
# Clear pytest cache
python -m pytest --cache-clear tests/

# Run with fresh imports
python -m pytest --forked tests/
```

#### Memory Issues
```bash
# Run tests with memory limit
python -m pytest tests/ --tb=no --maxfail=1

# Run subset of tests
python -m pytest tests/test_core/ tests/test_contracts/ -v
```

### Environment Debugging
```bash
# Show pytest version and plugins
python -m pytest --version

# Show test collection without running
python -m pytest tests/ --collect-only

# Show available fixtures
python -m pytest tests/ --fixtures

# Validate test discovery
python -m pytest tests/ --collect-only -q
```

## Best Practices

### During Development
1. **Start Small**: Test individual methods first
2. **Use Keywords**: Filter tests with `-k` for focused testing
3. **Quick Feedback**: Use `-x` to stop on first failure
4. **Clean Output**: Use `--tb=short` for readable errors

### Before Commits
1. **Full Suite**: Run all tests with coverage
2. **Check Warnings**: Address deprecation warnings
3. **Performance**: Monitor test execution time
4. **Coverage**: Maintain coverage standards

### Sample Development Workflow
```bash
# 1. Quick check during development
python -m pytest tests/test_orders/test_manager.py::TestBasicOrderPlacement::test_place_buy_order -v

# 2. Test related functionality
python -m pytest tests/ -k "place_order" -v --tb=short

# 3. Test full module
python -m pytest tests/test_orders/test_manager.py -v

# 4. Full test suite before commit
python -m pytest tests/ --cov=src --cov-report=term-missing -v
```

---

## Quick Reference Card

| Command | Purpose |
|---------|---------|
| `python -m pytest tests/ -v` | Run all tests |
| `python -m pytest tests/ -k "order" -v` | Filter by keyword |
| `python -m pytest tests/test_app/test_main.py -v` | Test specific module |
| `python -m pytest tests/ --cov=src` | Run with coverage |
| `python -m pytest tests/ -x` | Stop on first failure |
| `python -m pytest tests/ --pdb` | Debug failures |
| `python -m pytest tests/ -n auto` | Parallel execution |
| `python -m pytest tests/ --lf` | Re-run last failures |

This reference covers all the testing commands and patterns you'll need for the IBKR trading application development and maintenance.
