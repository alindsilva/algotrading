"""
Enhanced contract factory for creating IBKR contracts.
Provides comprehensive methods for stocks, options, futures, forex, and other instruments.
"""

from typing import Optional, Union, List
from datetime import datetime, date
from decimal import Decimal

from ibapi.contract import Contract, ContractDetails
from ibapi.order import Order

from ..core.types import SecurityType, Exchange, Currency
from ..core.exceptions import ValidationError


class ContractFactory:
    """Factory for creating various types of IBKR contracts"""
    
    @staticmethod
    def create_stock(symbol: str, 
                    exchange: Union[str, Exchange] = Exchange.SMART,
                    currency: Union[str, Currency] = Currency.USD,
                    primary_exchange: Optional[str] = None) -> Contract:
        """
        Create a stock contract
        
        Args:
            symbol: Stock symbol
            exchange: Exchange to trade on
            currency: Currency denomination
            primary_exchange: Primary exchange (optional)
        
        Returns:
            Stock contract
        """
        if not symbol or not isinstance(symbol, str):
            raise ValidationError("Symbol must be a non-empty string")
        
        contract = Contract()
        contract.symbol = symbol.upper()
        contract.secType = SecurityType.STOCK.value
        contract.exchange = exchange.value if isinstance(exchange, Exchange) else exchange
        contract.currency = currency.value if isinstance(currency, Currency) else currency
        
        if primary_exchange:
            contract.primaryExchange = primary_exchange
        else:
            contract.primaryExchange = None
        
        return contract
    
    @staticmethod
    def create_option(symbol: str,
                     expiry: Union[str, date, datetime],
                     strike: Union[float, Decimal],
                     right: str,
                     exchange: Union[str, Exchange] = Exchange.SMART,
                     currency: Union[str, Currency] = Currency.USD,
                     multiplier: str = "100") -> Contract:
        """
        Create an option contract
        
        Args:
            symbol: Underlying symbol
            expiry: Option expiration date (YYYYMMDD format or date object)
            strike: Strike price
            right: Option right ('C' for call, 'P' for put)
            exchange: Exchange to trade on
            currency: Currency denomination
            multiplier: Contract multiplier
        
        Returns:
            Option contract
        """
        if not symbol or not isinstance(symbol, str):
            raise ValidationError("Symbol must be a non-empty string")
        
        if right not in ['C', 'P', 'CALL', 'PUT']:
            raise ValidationError("Option right must be 'C', 'P', 'CALL', or 'PUT'")
        
        # Normalize right
        right = 'C' if right in ['C', 'CALL'] else 'P'
        
        # Handle expiry date formatting
        if isinstance(expiry, (date, datetime)):
            expiry_str = expiry.strftime("%Y%m%d")
        else:
            expiry_str = str(expiry)
        
        # Validate expiry format - check for invalid formats like "2023-12-15"
        if '-' in expiry_str or len(expiry_str) != 8 or not expiry_str.isdigit():
            raise ValidationError("Expiry must be in YYYYMMDD format")
        
        # Validate that it's a valid date
        try:
            year = int(expiry_str[:4])
            month = int(expiry_str[4:6])
            day = int(expiry_str[6:8])
            datetime(year, month, day)  # This will raise ValueError if invalid
        except ValueError:
            raise ValidationError("Expiry must be in YYYYMMDD format")
        
        contract = Contract()
        contract.symbol = symbol.upper()
        contract.secType = SecurityType.OPTION.value
        contract.exchange = exchange.value if isinstance(exchange, Exchange) else exchange
        contract.currency = currency.value if isinstance(currency, Currency) else currency
        contract.lastTradeDateOrContractMonth = expiry_str
        contract.strike = float(strike)
        contract.right = right
        contract.multiplier = multiplier
        
        return contract
    
    @staticmethod
    def create_future(symbol: str,
                     expiry: Union[str, date, datetime],
                     exchange: Union[str, Exchange],
                     currency: Union[str, Currency] = Currency.USD,
                     multiplier: Optional[str] = None) -> Contract:
        """
        Create a futures contract
        
        Args:
            symbol: Future symbol
            expiry: Contract expiration (YYYYMM format or date object)
            exchange: Exchange to trade on
            currency: Currency denomination
            multiplier: Contract multiplier
        
        Returns:
            Futures contract
        """
        if not symbol or not isinstance(symbol, str):
            raise ValidationError("Symbol must be a non-empty string")
        
        # Handle expiry date formatting for futures (YYYYMM)
        if isinstance(expiry, (date, datetime)):
            expiry_str = expiry.strftime("%Y%m")
        else:
            expiry_str = str(expiry)
        
        # Validate expiry format (should be YYYYMM for futures)
        if len(expiry_str) not in [6, 8]:
            raise ValidationError("Future expiry must be in YYYYMM or YYYYMMDD format")
        
        contract = Contract()
        contract.symbol = symbol.upper()
        contract.secType = SecurityType.FUTURE.value
        contract.exchange = exchange.value if isinstance(exchange, Exchange) else exchange
        contract.currency = currency.value if isinstance(currency, Currency) else currency
        contract.lastTradeDateOrContractMonth = expiry_str
        
        if multiplier:
            contract.multiplier = multiplier
        else:
            contract.multiplier = None
        
        return contract
    
    @staticmethod
    def create_forex(base_currency: Union[str, Currency],
                    quote_currency: Union[str, Currency],
                    exchange: Union[str, Exchange] = Exchange.IDEALPRO) -> Contract:
        """
        Create a forex contract
        
        Args:
            base_currency: Base currency (e.g., 'EUR')
            quote_currency: Quote currency (e.g., 'USD')
            exchange: Exchange (typically IDEALPRO for forex)
        
        Returns:
            Forex contract
        """
        base = base_currency.value if isinstance(base_currency, Currency) else str(base_currency).upper()
        quote = quote_currency.value if isinstance(quote_currency, Currency) else str(quote_currency).upper()
        
        if base == quote:
            raise ValidationError("Base and quote currencies cannot be the same")
        
        contract = Contract()
        contract.symbol = base
        contract.secType = SecurityType.CASH.value
        contract.currency = quote
        contract.exchange = exchange.value if isinstance(exchange, Exchange) else exchange
        
        return contract
    
    @staticmethod
    def create_index(symbol: str,
                    exchange: Union[str, Exchange],
                    currency: Union[str, Currency] = Currency.USD) -> Contract:
        """
        Create an index contract
        
        Args:
            symbol: Index symbol
            exchange: Exchange
            currency: Currency denomination
        
        Returns:
            Index contract
        """
        if not symbol or not isinstance(symbol, str):
            raise ValidationError("Symbol must be a non-empty string")
        
        contract = Contract()
        contract.symbol = symbol.upper()
        contract.secType = SecurityType.INDEX.value
        contract.exchange = exchange.value if isinstance(exchange, Exchange) else exchange
        contract.currency = currency.value if isinstance(currency, Currency) else currency
        
        return contract
    
    @staticmethod
    def create_etf(symbol: str,
                  exchange: Union[str, Exchange] = Exchange.SMART,
                  currency: Union[str, Currency] = Currency.USD,
                  primary_exchange: Optional[str] = None) -> Contract:
        """
        Create an ETF contract (treated as stock)
        
        Args:
            symbol: ETF symbol
            exchange: Exchange to trade on
            currency: Currency denomination
            primary_exchange: Primary exchange (optional)
        
        Returns:
            ETF contract
        """
        return ContractFactory.create_stock(symbol, exchange, currency, primary_exchange)
    
    @staticmethod
    def create_bond(symbol: str,
                   exchange: Union[str, Exchange],
                   currency: Union[str, Currency] = Currency.USD) -> Contract:
        """
        Create a bond contract
        
        Args:
            symbol: Bond symbol
            exchange: Exchange
            currency: Currency denomination
        
        Returns:
            Bond contract
        """
        if not symbol or not isinstance(symbol, str):
            raise ValidationError("Symbol must be a non-empty string")
        
        contract = Contract()
        contract.symbol = symbol.upper()
        contract.secType = SecurityType.BOND.value
        contract.exchange = exchange.value if isinstance(exchange, Exchange) else exchange
        contract.currency = currency.value if isinstance(currency, Currency) else currency
        
        return contract
    
    @staticmethod
    def create_commodity(symbol: str,
                        exchange: Union[str, Exchange],
                        currency: Union[str, Currency] = Currency.USD) -> Contract:
        """
        Create a commodity contract
        
        Args:
            symbol: Commodity symbol
            exchange: Exchange
            currency: Currency denomination
        
        Returns:
            Commodity contract
        """
        if not symbol or not isinstance(symbol, str):
            raise ValidationError("Symbol must be a non-empty string")
        
        contract = Contract()
        contract.symbol = symbol.upper()
        contract.secType = SecurityType.COMMODITY.value
        contract.exchange = exchange.value if isinstance(exchange, Exchange) else exchange
        contract.currency = currency.value if isinstance(currency, Currency) else currency
        
        return contract
    
    @staticmethod
    def create_crypto(symbol: str,
                     exchange: Union[str, Exchange] = "PAXOS",
                     currency: Union[str, Currency] = Currency.USD) -> Contract:
        """
        Create a cryptocurrency contract
        
        Args:
            symbol: Crypto symbol (e.g., 'BTC')
            exchange: Exchange (typically PAXOS)
            currency: Quote currency
        
        Returns:
            Cryptocurrency contract
        """
        if not symbol or not isinstance(symbol, str):
            raise ValidationError("Symbol must be a non-empty string")
        
        contract = Contract()
        contract.symbol = symbol.upper()
        contract.secType = SecurityType.CRYPTO.value
        contract.exchange = exchange.value if isinstance(exchange, Exchange) else exchange
        contract.currency = currency.value if isinstance(currency, Currency) else currency
        
        return contract
    
    @staticmethod
    def create_from_symbol(symbol_string: str) -> Contract:
        """
        Create a contract from a symbol string with automatic type detection
        
        Supports formats like:
        - "AAPL" -> Stock
        - "AAPL 20231215 C 150" -> Option
        - "ES 202312" -> Future
        - "EURUSD" -> Forex (if 6 chars and looks like currency pair)
        
        Args:
            symbol_string: Symbol string to parse
        
        Returns:
            Contract based on detected type
        """
        if not symbol_string or not isinstance(symbol_string, str):
            raise ValidationError("Symbol string must be a non-empty string")
        
        parts = symbol_string.strip().upper().split()
        
        if len(parts) == 1:
            symbol = parts[0]
            
            # Check if it looks like a forex pair (6 characters, common currency codes)
            if len(symbol) == 6:
                common_currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'AUD', 'CAD', 'NZD']
                base = symbol[:3]
                quote = symbol[3:]
                
                if base in common_currencies and quote in common_currencies:
                    return ContractFactory.create_forex(base, quote)
            
            # Default to stock
            return ContractFactory.create_stock(symbol)
        
        elif len(parts) == 4:
            # Assume option: SYMBOL EXPIRY RIGHT STRIKE
            symbol, expiry, right, strike = parts
            try:
                strike_value = float(strike)
            except ValueError:
                raise ValidationError(f"Cannot parse symbol string: {symbol_string}")
            return ContractFactory.create_option(
                symbol=symbol,
                expiry=expiry,
                strike=strike_value,
                right=right
            )
        
        elif len(parts) == 2:
            # Assume future: SYMBOL EXPIRY
            symbol, expiry = parts
            # Try to determine exchange based on symbol
            exchange = ContractFactory._get_future_exchange(symbol)
            return ContractFactory.create_future(
                symbol=symbol,
                expiry=expiry,
                exchange=exchange
            )
        
        else:
            raise ValidationError(f"Cannot parse symbol string: {symbol_string}")
    
    @staticmethod
    def _get_future_exchange(symbol: str) -> str:
        """Get typical exchange for common futures symbols"""
        future_exchanges = {
            'ES': 'CME',    # E-mini S&P 500
            'NQ': 'CME',    # E-mini Nasdaq
            'YM': 'CBOT',   # E-mini Dow
            'RTY': 'CME',   # E-mini Russell
            'CL': 'NYMEX',  # Crude Oil
            'NG': 'NYMEX',  # Natural Gas
            'GC': 'COMEX',  # Gold
            'SI': 'COMEX',  # Silver
            'ZB': 'CBOT',   # T-Bond
            'ZN': 'CBOT',   # 10-Year Note
            'ZF': 'CBOT',   # 5-Year Note
            'ZT': 'CBOT',   # 2-Year Note
        }
        
        return future_exchanges.get(symbol, 'CME')  # Default to CME
    
    @staticmethod
    def validate_contract(contract: Contract) -> bool:
        """
        Validate that a contract has required fields
        
        Args:
            contract: Contract to validate
        
        Returns:
            True if valid
        
        Raises:
            ValidationError: If contract is invalid
        """
        if not contract.symbol:
            raise ValidationError("Contract must have a symbol")
        
        if not contract.secType:
            raise ValidationError("Contract must have a security type")
        
        if not contract.exchange:
            raise ValidationError("Contract must have an exchange")
        
        if not contract.currency:
            raise ValidationError("Contract must have a currency")
        
        # Type-specific validations
        if contract.secType == SecurityType.OPTION.value:
            if not contract.lastTradeDateOrContractMonth:
                raise ValidationError("Option contract must have expiry date")
            if contract.strike <= 0:
                raise ValidationError("Option contract must have valid strike price")
            if contract.right not in ['C', 'P']:
                raise ValidationError("Option contract must have valid right (C or P)")
        
        elif contract.secType == SecurityType.FUTURE.value:
            if not contract.lastTradeDateOrContractMonth:
                raise ValidationError("Future contract must have expiry date")
        
        return True
    
    @staticmethod
    def contract_to_string(contract: Contract) -> str:
        """
        Convert a contract to a readable string representation
        
        Args:
            contract: Contract to convert
        
        Returns:
            String representation of the contract
        """
        base = f"{contract.symbol} {contract.secType} {contract.exchange} {contract.currency}"
        
        if contract.secType == SecurityType.OPTION.value:
            return f"{base} {contract.lastTradeDateOrContractMonth} {contract.right} {contract.strike}"
        
        elif contract.secType == SecurityType.FUTURE.value:
            return f"{base} {contract.lastTradeDateOrContractMonth}"
        
        elif contract.primaryExchange:
            return f"{base} (Primary: {contract.primaryExchange})"
        
        return base
    
    @staticmethod
    def create_contract_list(symbols: List[str], 
                           contract_type: SecurityType = SecurityType.STOCK,
                           exchange: Union[str, Exchange] = Exchange.SMART,
                           currency: Union[str, Currency] = Currency.USD) -> List[Contract]:
        """
        Create a list of contracts from symbols
        
        Args:
            symbols: List of symbol strings
            contract_type: Type of contracts to create
            exchange: Exchange for all contracts
            currency: Currency for all contracts
        
        Returns:
            List of contracts
        """
        contracts = []
        
        for symbol in symbols:
            if contract_type == SecurityType.STOCK:
                contract = ContractFactory.create_stock(symbol, exchange, currency)
            elif contract_type == SecurityType.ETF:
                contract = ContractFactory.create_etf(symbol, exchange, currency)
            else:
                raise ValidationError(f"Bulk creation not supported for {contract_type}")
            
            contracts.append(contract)
        
        return contracts
