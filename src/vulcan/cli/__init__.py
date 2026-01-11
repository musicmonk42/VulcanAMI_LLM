"""
Vulcan CLI Module

This module provides command-line interface and interactive mode functionality
for VULCAN-AGI system operations.

Available Components:
    - run_interactive_mode: Start interactive REPL interface

Example:
    from vulcan.cli import run_interactive_mode
    run_interactive_mode()
"""

__version__ = "1.0.0"
__author__ = "VULCAN-AGI Team"

from vulcan.cli.interactive import run_interactive_mode

__all__ = ["run_interactive_mode"]
