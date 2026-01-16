"""
Vulcan CLI Configuration Management

Manages CLI configuration with support for environment variables
and optional config file (~/.vulcan/config.yaml).
"""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class CLIConfig:
    """
    CLI configuration manager.
    
    Loads configuration from:
    1. Environment variables (highest priority)
    2. Config file (~/.vulcan/config.yaml) - if exists
    3. Defaults
    
    Environment Variables:
        VULCAN_SERVER_URL: Server URL (default: http://localhost:8000)
        VULCAN_API_KEY: API key for authentication
    """
    
    DEFAULT_SERVER_URL = "http://localhost:8000"
    CONFIG_DIR = Path.home() / ".vulcan"
    CONFIG_FILE = CONFIG_DIR / "config.yaml"
    
    def __init__(self):
        """Initialize configuration manager."""
        self.server_url: Optional[str] = None
        self.api_key: Optional[str] = None
        self._load_config()
    
    def _load_config(self):
        """Load configuration from environment and config file."""
        # 1. Load from environment (highest priority)
        self.server_url = os.environ.get("VULCAN_SERVER_URL")
        self.api_key = os.environ.get("VULCAN_API_KEY")
        
        # 2. Load from config file if exists and values not set
        if self.CONFIG_FILE.exists():
            try:
                import yaml
                with open(self.CONFIG_FILE, 'r') as f:
                    config_data = yaml.safe_load(f) or {}
                
                if not self.server_url and "server_url" in config_data:
                    self.server_url = config_data["server_url"]
                if not self.api_key and "api_key" in config_data:
                    self.api_key = config_data["api_key"]
                
                logger.debug(f"Loaded configuration from {self.CONFIG_FILE}")
            except ImportError:
                logger.debug("PyYAML not available, skipping config file")
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}")
        
        # 3. Apply defaults
        if not self.server_url:
            self.server_url = self.DEFAULT_SERVER_URL
    
    def save_config(self, server_url: Optional[str] = None, api_key: Optional[str] = None):
        """
        Save configuration to config file.
        
        Args:
            server_url: Server URL to save
            api_key: API key to save
        """
        try:
            import yaml
        except ImportError:
            logger.error("PyYAML not installed. Cannot save config file.")
            logger.info("Install with: pip install PyYAML")
            return
        
        # Create config directory if needed
        self.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        
        # Load existing config or create new
        config_data = {}
        if self.CONFIG_FILE.exists():
            try:
                with open(self.CONFIG_FILE, 'r') as f:
                    config_data = yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning(f"Failed to load existing config: {e}")
        
        # Update values
        if server_url is not None:
            config_data["server_url"] = server_url
            self.server_url = server_url
        if api_key is not None:
            config_data["api_key"] = api_key
            self.api_key = api_key
        
        # Save to file
        try:
            with open(self.CONFIG_FILE, 'w') as f:
                yaml.safe_dump(config_data, f, default_flow_style=False)
            
            # Set secure permissions (readable only by owner)
            self.CONFIG_FILE.chmod(0o600)
            
            logger.info(f"Configuration saved to {self.CONFIG_FILE}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def get_server_url(self) -> str:
        """Get configured server URL."""
        return self.server_url or self.DEFAULT_SERVER_URL
    
    def get_api_key(self) -> Optional[str]:
        """Get configured API key."""
        return self.api_key
    
    def __repr__(self) -> str:
        """String representation."""
        api_key_display = "***" if self.api_key else "None"
        return f"CLIConfig(server_url={self.server_url}, api_key={api_key_display})"
