"""
Vulcan CLI Module

This module provides command-line interface and interactive mode functionality
for VULCAN-AGI system operations.

Available Components:
    - run_interactive_mode: Start interactive REPL interface
    - VulcanClient: HTTP client for VULCAN API
    - CLIConfig: Configuration management

Example:
    from vulcan.cli import run_interactive_mode
    run_interactive_mode()
    
    # Or use the client directly
    from vulcan.cli import VulcanClient
    client = VulcanClient.from_settings()
    result = client.chat("Hello, VULCAN!")
"""

__version__ = "2.0.0"
__author__ = "VULCAN-AGI Team"

from vulcan.cli.client import VulcanAPIError, VulcanClient
from vulcan.cli.config import CLIConfig
from vulcan.cli.interactive import run_interactive_mode

__all__ = [
    "run_interactive_mode",
    "VulcanClient",
    "VulcanAPIError",
    "CLIConfig",
]
