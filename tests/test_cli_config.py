"""
Test suite for vulcan.cli.config - Configuration management

Tests for CLIConfig including environment variables, config file handling,
and secure defaults.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from vulcan.cli.config import CLIConfig


class TestCLIConfigInit:
    """Test CLIConfig initialization."""
    
    def test_init_defaults(self):
        """Test initialization with defaults."""
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(Path, 'exists', return_value=False):
                config = CLIConfig()
                assert config.server_url == CLIConfig.DEFAULT_SERVER_URL
                assert config.api_key is None
    
    def test_init_from_env(self):
        """Test initialization from environment variables."""
        with patch.dict(os.environ, {
            "VULCAN_SERVER_URL": "https://prod.example.com",
            "VULCAN_API_KEY": "test-key-123"
        }):
            with patch.object(Path, 'exists', return_value=False):
                config = CLIConfig()
                assert config.server_url == "https://prod.example.com"
                assert config.api_key == "test-key-123"
    
    def test_env_takes_priority_over_file(self):
        """Test that environment variables take priority over config file."""
        mock_yaml_data = {
            "server_url": "https://file.example.com",
            "api_key": "file-key"
        }
        
        with patch.dict(os.environ, {
            "VULCAN_SERVER_URL": "https://env.example.com",
            "VULCAN_API_KEY": "env-key"
        }):
            with patch.object(Path, 'exists', return_value=True):
                with patch("builtins.open", create=True):
                    with patch("yaml.safe_load", return_value=mock_yaml_data):
                        config = CLIConfig()
                        # Environment should win
                        assert config.server_url == "https://env.example.com"
                        assert config.api_key == "env-key"


class TestCLIConfigLoadFromFile:
    """Test loading configuration from file."""
    
    def test_load_from_yaml_file(self):
        """Test loading from YAML config file."""
        mock_yaml_data = {
            "server_url": "https://yaml.example.com",
            "api_key": "yaml-key-456"
        }
        
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(Path, 'exists', return_value=True):
                with patch("builtins.open", create=True):
                    with patch("yaml.safe_load", return_value=mock_yaml_data):
                        config = CLIConfig()
                        assert config.server_url == "https://yaml.example.com"
                        assert config.api_key == "yaml-key-456"
    
    def test_load_yaml_import_error(self):
        """Test graceful handling when PyYAML not available."""
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(Path, 'exists', return_value=True):
                with patch("builtins.open", create=True):
                    with patch("vulcan.cli.config.yaml", side_effect=ImportError):
                        # Should not raise, just skip file loading
                        config = CLIConfig()
                        assert config.server_url == CLIConfig.DEFAULT_SERVER_URL
    
    def test_load_invalid_yaml(self):
        """Test handling of invalid YAML file."""
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(Path, 'exists', return_value=True):
                with patch("builtins.open", create=True):
                    with patch("yaml.safe_load", side_effect=Exception("Invalid YAML")):
                        # Should not crash, just log warning and use defaults
                        config = CLIConfig()
                        assert config.server_url == CLIConfig.DEFAULT_SERVER_URL
    
    def test_load_empty_yaml(self):
        """Test handling of empty YAML file."""
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(Path, 'exists', return_value=True):
                with patch("builtins.open", create=True):
                    with patch("yaml.safe_load", return_value=None):
                        config = CLIConfig()
                        assert config.server_url == CLIConfig.DEFAULT_SERVER_URL


class TestCLIConfigSave:
    """Test saving configuration to file."""
    
    def test_save_config(self):
        """Test saving configuration to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".vulcan"
            config_file = config_dir / "config.yaml"
            
            with patch.object(CLIConfig, 'CONFIG_DIR', config_dir):
                with patch.object(CLIConfig, 'CONFIG_FILE', config_file):
                    config = CLIConfig()
                    
                    # Save configuration
                    config.save_config(
                        server_url="https://saved.example.com",
                        api_key="saved-key"
                    )
                    
                    # Verify directory was created
                    assert config_dir.exists()
                    
                    # Verify file was created
                    assert config_file.exists()
                    
                    # Verify file permissions (readable only by owner)
                    import stat
                    file_mode = config_file.stat().st_mode
                    assert stat.S_IMODE(file_mode) == 0o600
                    
                    # Verify values were updated
                    assert config.server_url == "https://saved.example.com"
                    assert config.api_key == "saved-key"
    
    def test_save_config_no_yaml(self):
        """Test save_config when PyYAML not available."""
        config = CLIConfig()
        
        with patch("vulcan.cli.config.yaml", side_effect=ImportError):
            # Should log error but not crash
            config.save_config(server_url="https://test.com")
    
    def test_save_config_partial_update(self):
        """Test saving only some config values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".vulcan"
            config_file = config_dir / "config.yaml"
            
            with patch.object(CLIConfig, 'CONFIG_DIR', config_dir):
                with patch.object(CLIConfig, 'CONFIG_FILE', config_file):
                    config = CLIConfig()
                    
                    # Save only server_url
                    config.save_config(server_url="https://new.example.com")
                    assert config.server_url == "https://new.example.com"
                    
                    # api_key should remain unchanged
                    assert config.api_key is None
    
    def test_save_config_merge_with_existing(self):
        """Test that save merges with existing config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".vulcan"
            config_file = config_dir / "config.yaml"
            config_dir.mkdir(parents=True, exist_ok=True)
            
            # Create existing config file
            import yaml
            with open(config_file, 'w') as f:
                yaml.safe_dump({
                    "server_url": "https://old.example.com",
                    "other_setting": "preserved"
                }, f)
            
            with patch.object(CLIConfig, 'CONFIG_DIR', config_dir):
                with patch.object(CLIConfig, 'CONFIG_FILE', config_file):
                    config = CLIConfig()
                    
                    # Update only api_key
                    config.save_config(api_key="new-key")
                    
                    # Verify merge happened
                    with open(config_file, 'r') as f:
                        saved_data = yaml.safe_load(f)
                    
                    assert saved_data["api_key"] == "new-key"
                    assert saved_data["server_url"] == "https://old.example.com"
                    assert saved_data["other_setting"] == "preserved"


class TestCLIConfigGetters:
    """Test getter methods."""
    
    def test_get_server_url(self):
        """Test get_server_url method."""
        with patch.dict(os.environ, {"VULCAN_SERVER_URL": "https://test.com"}):
            with patch.object(Path, 'exists', return_value=False):
                config = CLIConfig()
                assert config.get_server_url() == "https://test.com"
    
    def test_get_server_url_default(self):
        """Test get_server_url returns default when not set."""
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(Path, 'exists', return_value=False):
                config = CLIConfig()
                assert config.get_server_url() == CLIConfig.DEFAULT_SERVER_URL
    
    def test_get_api_key(self):
        """Test get_api_key method."""
        with patch.dict(os.environ, {"VULCAN_API_KEY": "test-key"}):
            with patch.object(Path, 'exists', return_value=False):
                config = CLIConfig()
                assert config.get_api_key() == "test-key"
    
    def test_get_api_key_none(self):
        """Test get_api_key returns None when not set."""
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(Path, 'exists', return_value=False):
                config = CLIConfig()
                assert config.get_api_key() is None


class TestCLIConfigRepr:
    """Test string representation."""
    
    def test_repr_with_api_key(self):
        """Test repr with API key (should be masked)."""
        with patch.dict(os.environ, {
            "VULCAN_SERVER_URL": "https://test.com",
            "VULCAN_API_KEY": "secret-key-12345"
        }):
            with patch.object(Path, 'exists', return_value=False):
                config = CLIConfig()
                repr_str = repr(config)
                
                # Should show server URL
                assert "https://test.com" in repr_str
                
                # Should NOT expose actual API key
                assert "secret-key-12345" not in repr_str
                
                # Should indicate API key is set
                assert "***" in repr_str
    
    def test_repr_without_api_key(self):
        """Test repr without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(Path, 'exists', return_value=False):
                config = CLIConfig()
                repr_str = repr(config)
                
                assert "None" in repr_str
                assert CLIConfig.DEFAULT_SERVER_URL in repr_str


class TestCLIConfigConstants:
    """Test configuration constants."""
    
    def test_default_server_url(self):
        """Test default server URL is localhost."""
        assert CLIConfig.DEFAULT_SERVER_URL == "http://localhost:8000"
    
    def test_config_dir(self):
        """Test config directory is in home."""
        assert ".vulcan" in str(CLIConfig.CONFIG_DIR)
    
    def test_config_file(self):
        """Test config file name."""
        assert CLIConfig.CONFIG_FILE.name == "config.yaml"
