"""
VULCAN Tools Package - LLM-callable tools for precise computation.

This package provides tools that the LLM can call directly via function
calling when it needs precise computation that it cannot do natively.

Design Philosophy:
    - Tools are adapters that wrap existing reasoning engines
    - LLM decides when to use tools (no regex routing needed)
    - Tools handle: formal proofs, exact math, hashes, etc.
    - LLM handles: natural language, reasoning, planning

Available Tools:
    - sat_solver: SAT/SMT solving for logical satisfiability
    - math_engine: Symbolic math computation with SymPy
    - hash_compute: Cryptographic hash and encoding operations

Usage:
    # Get all tools in OpenAI function calling format
    from vulcan.tools import get_tools_for_llm
    tools = get_tools_for_llm()
    
    # Execute a specific tool
    from vulcan.tools import execute_tool
    result = execute_tool("hash_compute", {"data": "hello", "algorithm": "sha256"})
    
    # Get individual tool instances
    from vulcan.tools import SATSolverTool, MathEngineTool, HashComputeTool
    tool = SATSolverTool()
    result = tool.execute(formula="P ∧ ¬P")

Architecture:
    Tools wrap existing reasoning engines from vulcan.reasoning:
    
    Tool              | Wraps
    ------------------|--------------------------------
    sat_solver        | vulcan.reasoning.symbolic.reasoner.SymbolicReasoner
    math_engine       | vulcan.reasoning.mathematical_computation.MathematicalComputationTool
    hash_compute      | vulcan.reasoning.cryptographic_engine.CryptographicEngine

Version History:
    1.0.0 - Initial implementation with SAT, math, and hash tools
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Type

from .base import (
    Tool,
    ToolCall,
    ToolInput,
    ToolOutput,
    ToolResult,
    ToolStatus,
)
from .sat_solver import SATSolverTool
from .math_engine import MathEngineTool
from .hash_compute import HashComputeTool

# Module metadata
__version__ = "1.0.0"
__author__ = "VULCAN-AGI Team"

__all__ = [
    # Base classes
    "Tool",
    "ToolInput",
    "ToolOutput",
    "ToolCall",
    "ToolResult",
    "ToolStatus",
    # Tool implementations
    "SATSolverTool",
    "MathEngineTool",
    "HashComputeTool",
    # Registry functions
    "get_tools_for_llm",
    "execute_tool",
    "get_tool_by_name",
    "get_all_tools",
    "VULCAN_TOOLS",
]

logger = logging.getLogger(__name__)


# =============================================================================
# TOOL REGISTRY
# =============================================================================

# Singleton tool instances (lazy initialized)
_tool_instances: Dict[str, Tool] = {}
_registry_lock = __import__("threading").Lock()


def _get_tool_instance(tool_class: Type[Tool]) -> Tool:
    """
    Get or create a singleton instance of a tool.
    
    Thread-safe lazy initialization of tool instances.
    
    Args:
        tool_class: The tool class to instantiate
        
    Returns:
        Singleton tool instance
    """
    class_name = tool_class.__name__
    
    if class_name not in _tool_instances:
        with _registry_lock:
            # Double-check pattern
            if class_name not in _tool_instances:
                try:
                    _tool_instances[class_name] = tool_class()
                    logger.debug(f"Initialized tool: {class_name}")
                except Exception as e:
                    logger.error(f"Failed to initialize {class_name}: {e}")
                    raise
    
    return _tool_instances[class_name]


# Tool classes registry
TOOL_CLASSES: List[Type[Tool]] = [
    SATSolverTool,
    MathEngineTool,
    HashComputeTool,
]


def get_all_tools() -> List[Tool]:
    """
    Get all available tool instances.
    
    Returns:
        List of initialized tool instances
    """
    return [_get_tool_instance(cls) for cls in TOOL_CLASSES]


def get_tool_by_name(name: str) -> Optional[Tool]:
    """
    Get a tool instance by name.
    
    Args:
        name: Tool name (e.g., "sat_solver", "math_engine")
        
    Returns:
        Tool instance if found, None otherwise
    """
    for tool_class in TOOL_CLASSES:
        tool = _get_tool_instance(tool_class)
        if tool.name == name:
            return tool
    return None


# Pre-instantiated tool registry for convenience
# Lazy property that returns tools
@property
def VULCAN_TOOLS() -> List[Tool]:
    """All available VULCAN tools."""
    return get_all_tools()


def get_tools_for_llm(
    include_unavailable: bool = False,
) -> List[Dict[str, Any]]:
    """
    Get tool definitions in OpenAI function calling format.
    
    Returns a list of tool definitions suitable for use in the 'tools'
    parameter of OpenAI's chat completion API.
    
    Args:
        include_unavailable: If True, include tools even if their
                            underlying engine is not available
                            
    Returns:
        List of tool definitions in OpenAI format:
        [
            {
                "type": "function",
                "function": {
                    "name": "sat_solver",
                    "description": "...",
                    "parameters": {...}
                }
            },
            ...
        ]
    
    Example:
        >>> tools = get_tools_for_llm()
        >>> response = openai.chat.completions.create(
        ...     model="gpt-4",
        ...     messages=[...],
        ...     tools=tools,
        ...     tool_choice="auto",
        ... )
    """
    result = []
    
    for tool in get_all_tools():
        if include_unavailable or tool.is_available:
            result.append(tool.to_openai_tool())
        else:
            logger.debug(f"Skipping unavailable tool: {tool.name}")
    
    return result


def execute_tool(
    name: str,
    arguments: Dict[str, Any],
) -> ToolOutput:
    """
    Execute a tool by name with given arguments.
    
    This is the main entry point for tool execution from the LLM's
    function calling response.
    
    Args:
        name: Tool name (e.g., "sat_solver", "math_engine")
        arguments: Dict of arguments matching the tool's parameters_schema
        
    Returns:
        ToolOutput with execution result
        
    Raises:
        ValueError: If tool name is not found
        
    Example:
        >>> result = execute_tool("hash_compute", {
        ...     "data": "Hello, World!",
        ...     "algorithm": "sha256"
        ... })
        >>> print(result.result["hash"])
        'dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f'
    """
    tool = get_tool_by_name(name)
    
    if tool is None:
        available = [t.name for t in get_all_tools()]
        raise ValueError(
            f"Unknown tool: {name}. Available tools: {available}"
        )
    
    if not tool.is_available:
        return ToolOutput.create_failure(
            error=f"Tool '{name}' is not available (missing dependencies)",
            computation_time_ms=0.0,
            status=ToolStatus.UNAVAILABLE,
        )
    
    logger.info(f"Executing tool: {name} with args: {list(arguments.keys())}")
    
    try:
        result = tool.execute(**arguments)
        logger.info(
            f"Tool {name} completed: success={result.success}, "
            f"time_ms={result.computation_time_ms:.2f}"
        )
        return result
        
    except Exception as e:
        logger.error(f"Tool {name} execution failed: {e}", exc_info=True)
        return ToolOutput.create_failure(
            error=f"Tool execution failed: {str(e)}",
            computation_time_ms=0.0,
            status=ToolStatus.FAILURE,
        )


def get_tool_stats() -> Dict[str, Dict[str, Any]]:
    """
    Get execution statistics for all tools.
    
    Returns:
        Dict mapping tool names to their statistics
    """
    return {
        tool.name: tool.get_stats()
        for tool in get_all_tools()
    }
