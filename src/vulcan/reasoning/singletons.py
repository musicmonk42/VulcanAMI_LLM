"""
Global singletons for expensive components.
Ensures components are created ONCE per process.

This module provides thread-safe singleton access to key reasoning components
to prevent repeated initialization that causes:
- "[ReasoningIntegration] Initialized" appearing multiple times
- "Tool Selector initialized with 5 tools" appearing many times  
- "Warm pool initialized with 5 tool pools" appearing repeatedly
"""

import logging
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Module-level singletons
_tool_selector: Optional[Any] = None
_reasoning_integration: Optional[Any] = None
_portfolio_executor: Optional[Any] = None
_singleton_lock = threading.Lock()


def get_tool_selector():
    """
    Get or create the global ToolSelector instance.
    
    Returns:
        ToolSelector instance (singleton).
    """
    global _tool_selector
    
    if _tool_selector is not None:
        logger.debug("[Singletons] Returning cached ToolSelector")
        return _tool_selector
    
    with _singleton_lock:
        if _tool_selector is not None:
            return _tool_selector
        
        logger.info("[Singletons] Creating global ToolSelector (ONCE)")
        try:
            from vulcan.reasoning.selection.tool_selector import ToolSelector
            _tool_selector = ToolSelector()
            logger.info("[Singletons] ✓ ToolSelector created and cached")
            return _tool_selector
        except ImportError as e:
            logger.warning(f"[Singletons] ToolSelector not available: {e}")
            return None
        except Exception as e:
            logger.error(f"[Singletons] Failed to create ToolSelector: {e}")
            return None


def get_reasoning_integration():
    """
    Get or create the global ReasoningIntegration instance.
    
    Returns:
        ReasoningIntegration instance (singleton).
    """
    global _reasoning_integration
    
    if _reasoning_integration is not None:
        logger.debug("[Singletons] Returning cached ReasoningIntegration")
        return _reasoning_integration
    
    with _singleton_lock:
        if _reasoning_integration is not None:
            return _reasoning_integration
        
        logger.info("[Singletons] Creating global ReasoningIntegration (ONCE)")
        try:
            from vulcan.reasoning.reasoning_integration import ReasoningIntegration
            _reasoning_integration = ReasoningIntegration()
            logger.info("[Singletons] ✓ ReasoningIntegration created and cached")
            return _reasoning_integration
        except ImportError as e:
            logger.warning(f"[Singletons] ReasoningIntegration not available: {e}")
            return None
        except Exception as e:
            logger.error(f"[Singletons] Failed to create ReasoningIntegration: {e}")
            return None


def get_portfolio_executor():
    """
    Get or create the global PortfolioExecutor instance.
    
    Returns:
        PortfolioExecutor instance (singleton).
    """
    global _portfolio_executor
    
    if _portfolio_executor is not None:
        logger.debug("[Singletons] Returning cached PortfolioExecutor")
        return _portfolio_executor
    
    with _singleton_lock:
        if _portfolio_executor is not None:
            return _portfolio_executor
        
        logger.info("[Singletons] Creating global PortfolioExecutor (ONCE)")
        try:
            from vulcan.reasoning.selection.portfolio_executor import PortfolioExecutor
            _portfolio_executor = PortfolioExecutor()
            logger.info("[Singletons] ✓ PortfolioExecutor created and cached")
            return _portfolio_executor
        except ImportError as e:
            logger.warning(f"[Singletons] PortfolioExecutor not available: {e}")
            return None
        except Exception as e:
            logger.error(f"[Singletons] Failed to create PortfolioExecutor: {e}")
            return None


def initialize_all() -> dict:
    """
    Initialize all singletons at startup.
    
    Returns:
        Dictionary of component names to initialization success status.
    """
    logger.info("[Singletons] Initializing all global components...")
    results = {}
    
    # Initialize components in order of dependency
    ts = get_tool_selector()
    results['tool_selector'] = ts is not None
    
    ri = get_reasoning_integration()
    results['reasoning_integration'] = ri is not None
    
    pe = get_portfolio_executor()
    results['portfolio_executor'] = pe is not None
    
    success_count = sum(1 for v in results.values() if v)
    logger.info(f"[Singletons] ✓ {success_count}/{len(results)} components initialized")
    
    return results


def reset_all() -> None:
    """
    Reset all singletons. Use with caution - primarily for testing.
    This will force components to be recreated on next access.
    """
    global _tool_selector, _reasoning_integration, _portfolio_executor
    
    with _singleton_lock:
        _tool_selector = None
        _reasoning_integration = None
        _portfolio_executor = None
        logger.info("[Singletons] All singletons reset")
