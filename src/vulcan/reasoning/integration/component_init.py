"""
Component initialization for reasoning integration.

Module: vulcan.reasoning.integration.component_init
Author: Vulcan AI Team
"""

import logging
from typing import Any, Dict, Optional

from .types import LOG_PREFIX

logger = logging.getLogger(__name__)


def init_components(config: Dict[str, Any]) -> Dict[str, Optional[Any]]:
    """
    Initialize all reasoning components.

    Args:
        config: Configuration dictionary
    
    Returns:
        Dictionary with component instances
    """
    return {
        'tool_selector': init_tool_selector(config),
        'portfolio_executor': init_portfolio_executor(config),
        'problem_decomposer': init_problem_decomposer(config),
        'query_bridge': init_query_bridge(config),
        'semantic_bridge': init_semantic_bridge(config),
        'domain_bridge': init_domain_bridge(config),
    }


def init_tool_selector(config: Dict[str, Any]) -> Optional[Any]:
    """
    Initialize ToolSelector component with error handling.

    PERFORMANCE FIX: Uses singleton from singletons.py to ensure ToolSelector
    is created exactly ONCE per process. This prevents progressive query routing
    degradation where each query creates new instances of:
    - WarmStartPool ("Warm pool initialized with 5 tool pools")
    - StochasticCostModel ("StochasticCostModel initialized")
    - BayesianMemoryPrior with SemanticToolMatcher

    Args:
        config: Configuration dictionary

    Returns:
        ToolSelector instance if successful, None otherwise.
    """
    try:
        # PERFORMANCE FIX: Use singleton instead of creating new instance
        # This prevents "Tool Selector initialized with 5 tools" appearing
        # multiple times and causing progressive routing time degradation
        from vulcan.reasoning.singletons import get_tool_selector

        selector = get_tool_selector()
        if selector is not None:
            logger.info(f"{LOG_PREFIX} ToolSelector obtained from singleton")
            return selector

        # Fallback: If singleton fails, try direct creation (should be rare)
        logger.warning(f"{LOG_PREFIX} Singleton unavailable, creating ToolSelector directly")
        from vulcan.reasoning.selection.tool_selector import ToolSelector
        selector = ToolSelector(config.get("tool_selector_config", {}))
        logger.info(f"{LOG_PREFIX} ToolSelector initialized successfully (fallback)")
        return selector

    except ImportError as e:
        logger.warning(
            f"{LOG_PREFIX} ToolSelector not available (missing dependency): {e}"
        )
    except Exception as e:
        logger.error(
            f"{LOG_PREFIX} ToolSelector initialization failed: {e}",
            exc_info=True
        )

    return None


def init_portfolio_executor(config: Dict[str, Any]) -> Optional[Any]:
    """
    Initialize PortfolioExecutor component with error handling.

    Args:
        config: Configuration dictionary

    Returns:
        PortfolioExecutor instance if successful, None otherwise.
    """
    try:
        from vulcan.reasoning.selection.portfolio_executor import PortfolioExecutor

        executor_config = config.get("portfolio_config", {})
        executor = PortfolioExecutor(executor_config)
        logger.info(f"{LOG_PREFIX} PortfolioExecutor initialized successfully")
        return executor

    except ImportError as e:
        logger.warning(
            f"{LOG_PREFIX} PortfolioExecutor not available (missing dependency): {e}"
        )
    except Exception as e:
        logger.error(
            f"{LOG_PREFIX} PortfolioExecutor initialization failed: {e}",
            exc_info=True
        )

    return None


def init_problem_decomposer(config: Dict[str, Any]) -> Optional[Any]:
    """
    Initialize ProblemDecomposer component with error handling.

    Args:
        config: Configuration dictionary

    Returns:
        ProblemDecomposer instance if successful, None otherwise.
    """
    try:
        from vulcan.reasoning.decomposition.problem_decomposer import ProblemDecomposer

        decomposer_config = config.get("decomposer_config", {})
        decomposer = ProblemDecomposer(decomposer_config)
        logger.info(f"{LOG_PREFIX} ProblemDecomposer initialized successfully")
        return decomposer

    except ImportError as e:
        logger.warning(
            f"{LOG_PREFIX} ProblemDecomposer not available (missing dependency): {e}"
        )
    except Exception as e:
        logger.error(
            f"{LOG_PREFIX} ProblemDecomposer initialization failed: {e}",
            exc_info=True
        )

    return None


def init_query_bridge(config: Dict[str, Any]) -> Optional[Any]:
    """
    Initialize QueryBridge component with error handling.

    Args:
        config: Configuration dictionary

    Returns:
        QueryBridge instance if successful, None otherwise.
    """
    try:
        from vulcan.reasoning.bridges.query_bridge import QueryBridge

        bridge = QueryBridge()
        logger.info(f"{LOG_PREFIX} QueryBridge initialized successfully")
        return bridge

    except ImportError as e:
        logger.warning(
            f"{LOG_PREFIX} QueryBridge not available (missing dependency): {e}"
        )
    except Exception as e:
        logger.error(
            f"{LOG_PREFIX} QueryBridge initialization failed: {e}",
            exc_info=True
        )

    return None


def init_semantic_bridge(config: Dict[str, Any]) -> Optional[Any]:
    """
    Initialize SemanticBridge component with error handling.

    Args:
        config: Configuration dictionary

    Returns:
        SemanticBridge instance if successful, None otherwise.
    """
    try:
        from vulcan.reasoning.bridges.semantic_bridge import SemanticBridge

        semantic_config = config.get("semantic_config", {})
        bridge = SemanticBridge(semantic_config)
        logger.info(f"{LOG_PREFIX} SemanticBridge initialized successfully")
        return bridge

    except ImportError as e:
        logger.warning(
            f"{LOG_PREFIX} SemanticBridge not available (missing dependency): {e}"
        )
    except Exception as e:
        logger.error(
            f"{LOG_PREFIX} SemanticBridge initialization failed: {e}",
            exc_info=True
        )

    return None


def init_domain_bridge(config: Dict[str, Any]) -> Optional[Any]:
    """
    Initialize DomainBridge component with error handling.

    Args:
        config: Configuration dictionary

    Returns:
        DomainBridge instance if successful, None otherwise.
    """
    try:
        from vulcan.reasoning.bridges.domain_bridge import DomainBridge

        domain_config = config.get("domain_config", {})
        bridge = DomainBridge(domain_config)
        logger.info(f"{LOG_PREFIX} DomainBridge initialized successfully")
        return bridge

    except ImportError as e:
        logger.warning(
            f"{LOG_PREFIX} DomainBridge not available (missing dependency): {e}"
        )
    except Exception as e:
        logger.error(
            f"{LOG_PREFIX} DomainBridge initialization failed: {e}",
            exc_info=True
        )

    return None


__all__ = [
    "init_components",
    "init_tool_selector",
    "init_portfolio_executor",
    "init_problem_decomposer",
    "init_query_bridge",
    "init_semantic_bridge",
    "init_domain_bridge",
]