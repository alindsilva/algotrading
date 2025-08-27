"""
Configuration management for IBKR trading application.
Provides centralized configuration with YAML and environment variable support.
"""

import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any, Dict

from .exceptions import ConfigurationError


@dataclass(frozen=True)
class DatabaseConfig:
    """Database configuration settings."""
    path: str = "data/trading_data.sqlite"
    pool_size: int = 5
    enable_wal_mode: bool = True
    
    def __post_init__(self):
        if self.pool_size <= 0:
            raise ValueError("Database pool size must be positive")


@dataclass(frozen=True)
class IBConfig:
    """IBKR API configuration settings."""
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 1
    connection_timeout: int = 30
    request_timeout: int = 10
    max_reconnect_attempts: int = 5
    reconnect_delay: float = 2.0
    risk_free_rate: float = 0.0
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    
    def __post_init__(self):
        # Validate port range
        if not (1024 <= self.port <= 65535):
            raise ValueError("Port must be between 1024 and 65535")
        
        # Validate timeouts
        if self.connection_timeout <= 0:
            raise ValueError("Connection timeout must be positive")
        
        if self.request_timeout <= 0:
            raise ValueError("Request timeout must be positive")
        
        # Validate reconnect attempts
        if self.max_reconnect_attempts <= 0:
            raise ValueError("Max reconnect attempts must be positive")
        
        # Validate reconnect delay
        if self.reconnect_delay <= 0.0:
            raise ValueError("Reconnect delay must be positive")
        
        # Validate risk-free rate
        if not (0.0 <= self.risk_free_rate <= 1.0):
            raise ValueError("Risk-free rate must be between 0 and 1")


def load_config(config_path: Optional[str] = None) -> IBConfig:
    """
    Load configuration from YAML file with environment variable overrides.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        IBConfig instance with loaded configuration
        
    Raises:
        ConfigurationError: If configuration file is invalid or missing
    """
    config_data = {}
    
    # Load from YAML file if provided
    if config_path:
        config_file = Path(config_path)
        if not config_file.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Failed to parse configuration file: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error reading configuration file: {e}")
    
    # Apply environment variable overrides
    _apply_env_overrides(config_data)
    
    try:
        # Create database config
        db_config_data = config_data.get('database', {})
        database_config = DatabaseConfig(
            path=db_config_data.get('path', "data/trading_data.sqlite"),
            pool_size=db_config_data.get('pool_size', 5),
            enable_wal_mode=db_config_data.get('enable_wal_mode', True)
        )
        
        # Create main config
        return IBConfig(
            host=config_data.get('host', "127.0.0.1"),
            port=config_data.get('port', 7497),                 # Adjust this to 7696 for production TWS API access
            client_id=config_data.get('client_id', 1),
            connection_timeout=config_data.get('connection_timeout', 30),
            request_timeout=config_data.get('request_timeout', 10),
            max_reconnect_attempts=config_data.get('max_reconnect_attempts', 5),
            reconnect_delay=config_data.get('reconnect_delay', 2.0),
            risk_free_rate=config_data.get('risk_free_rate', 0.0),
            database=database_config
        )
    except (ValueError, TypeError) as e:
        raise ConfigurationError(f"Configuration validation failed: {e}")


def _apply_env_overrides(config_data: Dict[str, Any]) -> None:
    """
    Apply environment variable overrides to configuration data.
    
    Args:
        config_data: Configuration dictionary to modify in-place
    """
    # IBKR connection settings
    ibkr_host = os.getenv('IBKR_HOST')
    if ibkr_host:
        config_data['host'] = ibkr_host
    
    ibkr_port = os.getenv('IBKR_PORT')
    if ibkr_port:
        try:
            config_data['port'] = int(ibkr_port)
        except ValueError:
            pass
    
    ibkr_client_id = os.getenv('IBKR_CLIENT_ID')
    if ibkr_client_id:
        try:
            config_data['client_id'] = int(ibkr_client_id)
        except ValueError:
            pass
    
    ibkr_risk_free_rate = os.getenv('IBKR_RISK_FREE_RATE')
    if ibkr_risk_free_rate:
        try:
            config_data['risk_free_rate'] = float(ibkr_risk_free_rate)
        except ValueError:
            pass
    
    # Database settings
    ibkr_db_path = os.getenv('IBKR_DATABASE_PATH')
    if ibkr_db_path:
        if 'database' not in config_data:
            config_data['database'] = {}
        config_data['database']['path'] = ibkr_db_path
