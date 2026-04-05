"""
Tool Wrapper Classes for VULCAN Tool Selection System.

These wrappers adapt the different reasoning engine interfaces to a common
reason(problem) interface expected by PortfolioExecutor._run_tool().

Extracted from tool_selector.py to reduce module size.
"""

from .causal import CausalToolWrapper
from .analogical import AnalogicalToolWrapper
from .multimodal import MultimodalToolWrapper
from .philosophical import PhilosophicalToolWrapper
from .cryptographic import CryptographicToolWrapper
from .symbolic import SymbolicToolWrapper
from .probabilistic import ProbabilisticToolWrapper
from .world_model_queries import WorldModelToolWrapper
from .mathematical import MathematicalToolWrapper

__all__ = [
    "CausalToolWrapper",
    "AnalogicalToolWrapper",
    "MultimodalToolWrapper",
    "PhilosophicalToolWrapper",
    "CryptographicToolWrapper",
    "SymbolicToolWrapper",
    "ProbabilisticToolWrapper",
    "WorldModelToolWrapper",
    "MathematicalToolWrapper",
]
