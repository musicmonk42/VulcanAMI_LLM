"""
Pytest configuration for VULCAN tests.

This conftest.py ensures the src directory is in the Python path
so that `from vulcan.xxx import yyy` style imports work correctly.
"""

import sys
import pathlib

# Add src directory to Python path
ROOT = pathlib.Path(__file__).resolve().parents[3]  # Go up to repo root
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
