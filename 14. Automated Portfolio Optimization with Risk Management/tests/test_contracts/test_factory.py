"""
Unit tests for the contract factory module.
"""

import pytest
from datetime import date, datetime
from decimal import Decimal

from src.contracts.factory import ContractFactory
from src.core.types import SecurityType, Exchange, Currency
from src.core.exceptions import ValidationError


class TestStockContracts:
    """Test stock contract creation."""
    
    def test_create_stock_basic(self):
        """Test basic stock contract creation."""
        contract = ContractFactory.create_stock("AAPL")
        
        assert contract.symbol == "AAPL"
        assert contract.secType == SecurityType.STOCK.value
        assert contract.exchange == Exchange.SMART.value
        assert contract.currency == Currency.USD.value
        assert contract.primaryExchange is None
    
    def test_create_stock_with_parameters(self):
        """Test stock contract creation with custom parameters."""
        contract = ContractFactory.create_stock(
            symbol="msft",
            exchange="NASDAQ",
            currency="USD",
            primary_exchange="NASDAQ"
        )
        
        assert contract.symbol == "MSFT"  # Should be uppercased
        assert contract.secType == SecurityType.STOCK.value
        assert contract.exchange == "NASDAQ"
        assert contract.currency == "USD"
        assert contract.primaryExchange == "NASDAQ"
    
    def test_create_stock_with_enums(self):
        """Test stock contract creation with enum parameters."""
        contract = ContractFactory.create_stock(
            symbol="googl",
            exchange=Exchange.NASDAQ,
            currency=Currency.USD
        )
        
        assert contract.symbol == "GOOGL"
        assert contract.exchange == Exchange.NASDAQ.value
        assert contract.currency == Currency.USD.value
    
    def test_create_stock_invalid_symbol(self):
        """Test stock contract creation with invalid symbol."""
        with pytest.raises(ValidationError, match="Symbol must be a non-empty string"):
            ContractFactory.create_stock("")
        
        with pytest.raises(ValidationError, match="Symbol must be a non-empty string"):
            ContractFactory.create_stock(None)


class TestOptionContracts:
    """Test option contract creation."""
    
    def test_create_option_basic(self):
        """Test basic option contract creation."""
        contract = ContractFactory.create_option(
            symbol="AAPL",
            expiry="20231215",
            strike=150.0,
            right="C"
        )
        
        assert contract.symbol == "AAPL"
        assert contract.secType == SecurityType.OPTION.value
        assert contract.exchange == Exchange.SMART.value
        assert contract.currency == Currency.USD.value
        assert contract.lastTradeDateOrContractMonth == "20231215"
        assert contract.strike == 150.0
        assert contract.right == "C"
        assert contract.multiplier == "100"
    
    def test_create_option_with_date_object(self):
        """Test option contract creation with date object."""
        expiry_date = date(2023, 12, 15)
        
        contract = ContractFactory.create_option(
            symbol="AAPL",
            expiry=expiry_date,
            strike=150.0,
            right="CALL"
        )
        
        assert contract.lastTradeDateOrContractMonth == "20231215"
        assert contract.right == "C"
    
    def test_create_option_with_datetime_object(self):
        """Test option contract creation with datetime object."""
        expiry_datetime = datetime(2023, 12, 15, 16, 0, 0)
        
        contract = ContractFactory.create_option(
            symbol="AAPL",
            expiry=expiry_datetime,
            strike=150.0,
            right="PUT"
        )
        
        assert contract.lastTradeDateOrContractMonth == "20231215"
        assert contract.right == "P"
    
    def test_create_option_with_decimal_strike(self):
        """Test option contract creation with Decimal strike price."""
        contract = ContractFactory.create_option(
            symbol="AAPL",
            expiry="20231215",
            strike=Decimal("150.50"),
            right="C"
        )
        
        assert contract.strike == 150.50
    
    def test_create_option_custom_multiplier(self):
        """Test option contract creation with custom multiplier."""
        contract = ContractFactory.create_option(
            symbol="AAPL",
            expiry="20231215",
            strike=150.0,
            right="C",
            multiplier="10"
        )
        
        assert contract.multiplier == "10"
    
    def test_create_option_invalid_expiry(self):
        """Test option contract creation with invalid expiry."""
        with pytest.raises(ValidationError, match="Expiry must be in YYYYMMDD format"):
            ContractFactory.create_option("AAPL", "2023-12-15", 150.0, "C")
        
        with pytest.raises(ValidationError, match="Expiry must be in YYYYMMDD format"):
            ContractFactory.create_option("AAPL", "20231232", 150.0, "C")  # Invalid date
    
    def test_create_option_invalid_right(self):
        """Test option contract creation with invalid right."""
        with pytest.raises(ValidationError, match="Option right must be"):
            ContractFactory.create_option("AAPL", "20231215", 150.0, "X")


class TestFutureContracts:
    """Test future contract creation."""
    
    def test_create_future_basic(self):
        """Test basic future contract creation."""
        contract = ContractFactory.create_future(
            symbol="ES",
            expiry="202312",
            exchange="CME"
        )
        
        assert contract.symbol == "ES"
        assert contract.secType == SecurityType.FUTURE.value
        assert contract.exchange == "CME"
        assert contract.currency == Currency.USD.value
        assert contract.lastTradeDateOrContractMonth == "202312"
        assert contract.multiplier is None
    
    def test_create_future_with_date(self):
        """Test future contract creation with date object."""
        expiry_date = date(2023, 12, 15)
        
        contract = ContractFactory.create_future(
            symbol="ES",
            expiry=expiry_date,
            exchange="CME"
        )
        
        assert contract.lastTradeDateOrContractMonth == "202312"  # Should format as YYYYMM
    
    def test_create_future_with_multiplier(self):
        """Test future contract creation with multiplier."""
        contract = ContractFactory.create_future(
            symbol="ES",
            expiry="202312",
            exchange="CME",
            multiplier="50"
        )
        
        assert contract.multiplier == "50"
    
    def test_create_future_invalid_expiry(self):
        """Test future contract creation with invalid expiry."""
        with pytest.raises(ValidationError, match="Future expiry must be in YYYYMM or YYYYMMDD format"):
            ContractFactory.create_future("ES", "2023", "CME")


class TestForexContracts:
    """Test forex contract creation."""
    
    def test_create_forex_basic(self):
        """Test basic forex contract creation."""
        contract = ContractFactory.create_forex("EUR", "USD")
        
        assert contract.symbol == "EUR"
        assert contract.secType == SecurityType.CASH.value
        assert contract.currency == "USD"
        assert contract.exchange == Exchange.IDEALPRO.value
    
    def test_create_forex_with_enums(self):
        """Test forex contract creation with currency enums."""
        contract = ContractFactory.create_forex(Currency.GBP, Currency.USD)
        
        assert contract.symbol == "GBP"
        assert contract.currency == "USD"
    
    def test_create_forex_custom_exchange(self):
        """Test forex contract creation with custom exchange."""
        contract = ContractFactory.create_forex("EUR", "USD", "FXCONV")
        
        assert contract.exchange == "FXCONV"
    
    def test_create_forex_same_currencies(self):
        """Test forex contract creation with same base and quote currencies."""
        with pytest.raises(ValidationError, match="Base and quote currencies cannot be the same"):
            ContractFactory.create_forex("USD", "USD")


class TestOtherContracts:
    """Test creation of other contract types."""
    
    def test_create_index(self):
        """Test index contract creation."""
        contract = ContractFactory.create_index("SPX", "CBOE")
        
        assert contract.symbol == "SPX"
        assert contract.secType == SecurityType.INDEX.value
        assert contract.exchange == "CBOE"
        assert contract.currency == Currency.USD.value
    
    def test_create_etf(self):
        """Test ETF contract creation."""
        contract = ContractFactory.create_etf("SPY")
        
        assert contract.symbol == "SPY"
        assert contract.secType == SecurityType.STOCK.value  # ETF is treated as stock
        assert contract.exchange == Exchange.SMART.value
    
    def test_create_bond(self):
        """Test bond contract creation."""
        contract = ContractFactory.create_bond("T", "BOND")
        
        assert contract.symbol == "T"
        assert contract.secType == SecurityType.BOND.value
        assert contract.exchange == "BOND"
    
    def test_create_commodity(self):
        """Test commodity contract creation."""
        contract = ContractFactory.create_commodity("GC", "COMEX")
        
        assert contract.symbol == "GC"
        assert contract.secType == SecurityType.COMMODITY.value
        assert contract.exchange == "COMEX"
    
    def test_create_crypto(self):
        """Test cryptocurrency contract creation."""
        contract = ContractFactory.create_crypto("BTC")
        
        assert contract.symbol == "BTC"
        assert contract.secType == SecurityType.CRYPTO.value
        assert contract.exchange == "PAXOS"
        assert contract.currency == Currency.USD.value


class TestContractParsing:
    """Test contract creation from string parsing."""
    
    def test_create_from_symbol_stock(self):
        """Test creating stock contract from symbol string."""
        contract = ContractFactory.create_from_symbol("AAPL")
        
        assert contract.symbol == "AAPL"
        assert contract.secType == SecurityType.STOCK.value
    
    def test_create_from_symbol_forex(self):
        """Test creating forex contract from symbol string."""
        contract = ContractFactory.create_from_symbol("EURUSD")
        
        assert contract.symbol == "EUR"
        assert contract.currency == "USD"
        assert contract.secType == SecurityType.CASH.value
    
    def test_create_from_symbol_option(self):
        """Test creating option contract from symbol string."""
        contract = ContractFactory.create_from_symbol("AAPL 20231215 C 150")
        
        assert contract.symbol == "AAPL"
        assert contract.lastTradeDateOrContractMonth == "20231215"
        assert contract.right == "C"
        assert contract.strike == 150.0
        assert contract.secType == SecurityType.OPTION.value
    
    def test_create_from_symbol_future(self):
        """Test creating future contract from symbol string."""
        contract = ContractFactory.create_from_symbol("ES 202312")
        
        assert contract.symbol == "ES"
        assert contract.lastTradeDateOrContractMonth == "202312"
        assert contract.secType == SecurityType.FUTURE.value
        assert contract.exchange == "CME"  # Should map ES to CME
    
    def test_create_from_symbol_invalid(self):
        """Test creating contract from invalid symbol string."""
        with pytest.raises(ValidationError, match="Cannot parse symbol string"):
            ContractFactory.create_from_symbol("INVALID TOO MANY PARTS")
        
        with pytest.raises(ValidationError):
            ContractFactory.create_from_symbol("")


class TestContractValidation:
    """Test contract validation functionality."""
    
    def test_validate_contract_valid_stock(self):
        """Test validation of valid stock contract."""
        contract = ContractFactory.create_stock("AAPL")
        assert ContractFactory.validate_contract(contract) is True
    
    def test_validate_contract_valid_option(self):
        """Test validation of valid option contract."""
        contract = ContractFactory.create_option("AAPL", "20231215", 150.0, "C")
        assert ContractFactory.validate_contract(contract) is True
    
    def test_validate_contract_missing_symbol(self):
        """Test validation of contract with missing symbol."""
        contract = ContractFactory.create_stock("AAPL")
        contract.symbol = ""
        
        with pytest.raises(ValidationError, match="Contract must have a symbol"):
            ContractFactory.validate_contract(contract)
    
    def test_validate_contract_invalid_option(self):
        """Test validation of invalid option contract."""
        contract = ContractFactory.create_option("AAPL", "20231215", 150.0, "C")
        contract.strike = 0  # Invalid strike price
        
        with pytest.raises(ValidationError, match="Option contract must have valid strike price"):
            ContractFactory.validate_contract(contract)
    
    def test_validate_contract_invalid_future(self):
        """Test validation of invalid future contract."""
        contract = ContractFactory.create_future("ES", "202312", "CME")
        contract.lastTradeDateOrContractMonth = ""  # Invalid expiry
        
        with pytest.raises(ValidationError, match="Future contract must have expiry date"):
            ContractFactory.validate_contract(contract)


class TestContractUtilities:
    """Test contract utility functions."""
    
    def test_contract_to_string_stock(self):
        """Test string representation of stock contract."""
        contract = ContractFactory.create_stock("AAPL", primary_exchange="NASDAQ")
        string_repr = ContractFactory.contract_to_string(contract)
        
        assert "AAPL" in string_repr
        assert "STK" in string_repr
        assert "NASDAQ" in string_repr
    
    def test_contract_to_string_option(self):
        """Test string representation of option contract."""
        contract = ContractFactory.create_option("AAPL", "20231215", 150.0, "C")
        string_repr = ContractFactory.contract_to_string(contract)
        
        assert "AAPL" in string_repr
        assert "20231215" in string_repr
        assert "C" in string_repr
        assert "150" in string_repr
    
    def test_create_contract_list(self):
        """Test creating list of contracts from symbols."""
        symbols = ["AAPL", "GOOGL", "MSFT"]
        contracts = ContractFactory.create_contract_list(
            symbols,
            SecurityType.STOCK,
            Exchange.NASDAQ,
            Currency.USD
        )
        
        assert len(contracts) == 3
        assert all(contract.secType == SecurityType.STOCK.value for contract in contracts)
        assert all(contract.exchange == Exchange.NASDAQ.value for contract in contracts)
        
        symbols_from_contracts = [contract.symbol for contract in contracts]
        assert "AAPL" in symbols_from_contracts
        assert "GOOGL" in symbols_from_contracts
        assert "MSFT" in symbols_from_contracts
    
    def test_create_contract_list_unsupported_type(self):
        """Test creating list of contracts with unsupported type."""
        with pytest.raises(ValidationError, match="Bulk creation not supported"):
            ContractFactory.create_contract_list(
                ["AAPL"],
                SecurityType.OPTION,  # Not supported for bulk creation
                Exchange.SMART
            )


class TestContractFactoryIntegration:
    """Integration tests for contract factory."""
    
    def test_comprehensive_contract_creation(self):
        """Test creating various types of contracts."""
        # Create different contract types
        stock = ContractFactory.create_stock("AAPL")
        option = ContractFactory.create_option("AAPL", "20231215", 150.0, "C")
        future = ContractFactory.create_future("ES", "202312", "CME")
        forex = ContractFactory.create_forex("EUR", "USD")
        
        # Validate all contracts
        assert ContractFactory.validate_contract(stock)
        assert ContractFactory.validate_contract(option)
        assert ContractFactory.validate_contract(future)
        assert ContractFactory.validate_contract(forex)
        
        # Test string representations
        stock_str = ContractFactory.contract_to_string(stock)
        option_str = ContractFactory.contract_to_string(option)
        future_str = ContractFactory.contract_to_string(future)
        forex_str = ContractFactory.contract_to_string(forex)
        
        assert all(len(s) > 0 for s in [stock_str, option_str, future_str, forex_str])
    
    def test_contract_factory_edge_cases(self):
        """Test contract factory with edge cases."""
        # Test with lowercase symbols
        contract = ContractFactory.create_stock("aapl")
        assert contract.symbol == "AAPL"
        
        # Test with mixed case exchange
        contract = ContractFactory.create_stock("AAPL", exchange="nasdaq")
        assert contract.exchange == "nasdaq"  # Exchange case is preserved
        
        # Test option with edge strike prices
        contract = ContractFactory.create_option("AAPL", "20231215", 0.01, "C")
        assert contract.strike == 0.01
        
        # Test future exchange mapping
        contracts = [
            ContractFactory.create_from_symbol("ES 202312"),
            ContractFactory.create_from_symbol("NQ 202312"),
            ContractFactory.create_from_symbol("GC 202312"),
        ]
        
        assert contracts[0].exchange == "CME"
        assert contracts[1].exchange == "CME" 
        assert contracts[2].exchange == "COMEX"
