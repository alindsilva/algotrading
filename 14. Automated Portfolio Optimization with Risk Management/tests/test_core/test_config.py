"""
Unit tests for the core configuration module.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open

from src.core.config import IBConfig, DatabaseConfig, load_config
from src.core.exceptions import ConfigurationError


class TestDatabaseConfig:
    """Test DatabaseConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = DatabaseConfig()
        assert config.path == "data/trading_data.sqlite"
        assert config.pool_size == 5
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = DatabaseConfig(
            path="/custom/path.db",
            pool_size=10
        )
        assert config.path == "/custom/path.db"
        assert config.pool_size == 10
    
    def test_validation(self):
        """Test configuration validation."""
        # Test invalid pool size
        with pytest.raises(ValueError):
            DatabaseConfig(pool_size=0)
        
        with pytest.raises(ValueError):
            DatabaseConfig(pool_size=-1)


class TestIBConfig:
    """Test IBConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = IBConfig()
        assert config.host == "127.0.0.1"
        assert config.port == 7497
        assert config.client_id == 1
        assert config.connection_timeout == 30
        assert config.request_timeout == 10
        assert config.max_reconnect_attempts == 5
        assert config.reconnect_delay == 2.0
        assert config.risk_free_rate == 0.0
        assert isinstance(config.database, DatabaseConfig)
    
    def test_custom_values(self):
        """Test custom configuration values."""
        db_config = DatabaseConfig(path="/test/db.sqlite", pool_size=3)
        
        config = IBConfig(
            host="192.168.1.100",
            port=7496,
            client_id=10,
            connection_timeout=60,
            request_timeout=20,
            max_reconnect_attempts=3,
            reconnect_delay=5.0,
            risk_free_rate=0.03,
            database=db_config
        )
        
        assert config.host == "192.168.1.100"
        assert config.port == 7496
        assert config.client_id == 10
        assert config.connection_timeout == 60
        assert config.request_timeout == 20
        assert config.max_reconnect_attempts == 3
        assert config.reconnect_delay == 5.0
        assert config.risk_free_rate == 0.03
        assert config.database.path == "/test/db.sqlite"
        assert config.database.pool_size == 3
    
    def test_validation(self):
        """Test configuration validation."""
        # Test invalid port
        with pytest.raises(ValueError):
            IBConfig(port=0)
        
        with pytest.raises(ValueError):
            IBConfig(port=100000)
        
        # Test invalid timeouts
        with pytest.raises(ValueError):
            IBConfig(connection_timeout=0)
        
        with pytest.raises(ValueError):
            IBConfig(request_timeout=0)
        
        # Test invalid reconnect attempts
        with pytest.raises(ValueError):
            IBConfig(max_reconnect_attempts=0)
        
        # Test invalid reconnect delay
        with pytest.raises(ValueError):
            IBConfig(reconnect_delay=0.0)
        
        # Test invalid risk-free rate
        with pytest.raises(ValueError):
            IBConfig(risk_free_rate=-1.0)
        
        with pytest.raises(ValueError):
            IBConfig(risk_free_rate=2.0)  # 200% risk-free rate is unrealistic


class TestLoadConfig:
    """Test configuration loading functionality."""
    
    def test_load_config_from_yaml_file(self, temp_dir):
        """Test loading configuration from a YAML file."""
        config_data = {
            'host': '192.168.1.100',
            'port': 7496,
            'client_id': 5,
            'connection_timeout': 45,
            'request_timeout': 15,
            'max_reconnect_attempts': 3,
            'reconnect_delay': 3.0,
            'risk_free_rate': 0.025,
            'database': {
                'path': '/custom/path.sqlite',
                'pool_size': 8
            }
        }
        
        config_file = temp_dir / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        config = load_config(str(config_file))
        
        assert config.host == '192.168.1.100'
        assert config.port == 7496
        assert config.client_id == 5
        assert config.connection_timeout == 45
        assert config.request_timeout == 15
        assert config.max_reconnect_attempts == 3
        assert config.reconnect_delay == 3.0
        assert config.risk_free_rate == 0.025
        assert config.database.path == '/custom/path.sqlite'
        assert config.database.pool_size == 8
    
    def test_load_config_nonexistent_file(self):
        """Test loading configuration from a nonexistent file."""
        with pytest.raises(ConfigurationError, match="Configuration file not found"):
            load_config("/nonexistent/config.yaml")
    
    def test_load_config_invalid_yaml(self, temp_dir):
        """Test loading configuration from invalid YAML file."""
        config_file = temp_dir / "invalid_config.yaml"
        with open(config_file, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        with pytest.raises(ConfigurationError, match="Failed to parse configuration"):
            load_config(str(config_file))
    
    def test_load_config_invalid_values(self, temp_dir):
        """Test loading configuration with invalid values."""
        config_data = {
            'host': '192.168.1.100',
            'port': -1,  # Invalid port
            'client_id': 5
        }
        
        config_file = temp_dir / "invalid_values_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        with pytest.raises(ConfigurationError, match="Configuration validation failed"):
            load_config(str(config_file))
    
    def test_load_config_partial_config(self, temp_dir):
        """Test loading partial configuration (should use defaults for missing values)."""
        config_data = {
            'host': 'custom_host',
            'port': 7496,
            'database': {
                'path': '/custom/db.sqlite'
                # pool_size missing, should use default
            }
        }
        
        config_file = temp_dir / "partial_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        config = load_config(str(config_file))
        
        assert config.host == 'custom_host'
        assert config.port == 7496
        assert config.client_id == 1  # Default value
        assert config.database.path == '/custom/db.sqlite'
        assert config.database.pool_size == 5  # Default value
    
    @patch.dict('os.environ', {
        'IBKR_HOST': '10.0.0.1',
        'IBKR_PORT': '7498',
        'IBKR_CLIENT_ID': '42',
        'IBKR_RISK_FREE_RATE': '0.035'
    })
    def test_environment_variable_override(self, temp_dir):
        """Test environment variable overrides."""
        config_data = {
            'host': '192.168.1.100',
            'port': 7496,
            'client_id': 5,
            'risk_free_rate': 0.02
        }
        
        config_file = temp_dir / "env_test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        config = load_config(str(config_file))
        
        # Environment variables should override config file values
        assert config.host == '10.0.0.1'
        assert config.port == 7498
        assert config.client_id == 42
        assert config.risk_free_rate == 0.035
    
    @patch.dict('os.environ', {'IBKR_DATABASE_PATH': '/env/database.sqlite'})
    def test_nested_environment_override(self, temp_dir):
        """Test environment variable overrides for nested config."""
        config_data = {
            'database': {
                'path': '/config/database.sqlite',
                'pool_size': 5
            }
        }
        
        config_file = temp_dir / "nested_env_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        config = load_config(str(config_file))
        
        assert config.database.path == '/env/database.sqlite'
        assert config.database.pool_size == 5  # Should remain unchanged
    
    def test_config_repr(self):
        """Test configuration string representation."""
        config = IBConfig(host="test_host", port=7498)
        repr_str = repr(config)
        
        assert "IBConfig" in repr_str
        assert "host='test_host'" in repr_str
        assert "port=7498" in repr_str
        # Should not contain sensitive information
        assert "password" not in repr_str.lower()
    
    def test_config_equality(self):
        """Test configuration equality comparison."""
        config1 = IBConfig(host="test", port=7497)
        config2 = IBConfig(host="test", port=7497)
        config3 = IBConfig(host="test", port=7498)
        
        assert config1 == config2
        assert config1 != config3
    
    def test_config_immutability(self):
        """Test that configuration objects are immutable."""
        config = IBConfig()
        
        # Should not be able to modify after creation (dataclass frozen=True)
        with pytest.raises(AttributeError):
            config.host = "new_host"


class TestConfigurationIntegration:
    """Integration tests for configuration functionality."""
    
    def test_full_configuration_lifecycle(self, temp_dir):
        """Test complete configuration loading and usage lifecycle."""
        # Create a comprehensive configuration file
        config_data = {
            'host': 'localhost',
            'port': 7497,
            'client_id': 100,
            'connection_timeout': 30,
            'request_timeout': 10,
            'max_reconnect_attempts': 5,
            'reconnect_delay': 2.0,
            'risk_free_rate': 0.02,
            'database': {
                'path': str(temp_dir / 'test_trading.sqlite'),
                'pool_size': 5
            }
        }
        
        config_file = temp_dir / "full_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Load configuration
        config = load_config(str(config_file))
        
        # Verify all values are correct
        assert config.host == 'localhost'
        assert config.port == 7497
        assert config.client_id == 100
        assert config.connection_timeout == 30
        assert config.request_timeout == 10
        assert config.max_reconnect_attempts == 5
        assert config.reconnect_delay == 2.0
        assert config.risk_free_rate == 0.02
        assert config.database.path == str(temp_dir / 'test_trading.sqlite')
        assert config.database.pool_size == 5
        
        # Test that config can be used in practice
        assert isinstance(config.database, DatabaseConfig)
        assert config.port > 1024  # Valid port range
        assert 0 <= config.risk_free_rate <= 1  # Valid risk-free rate
