"""
Central singleton registry for reasoning components.
All components that load ML models MUST use this registry.

This module provides thread-safe singleton access to key reasoning components
to prevent repeated initialization that causes:
- "[ReasoningIntegration] Initialized" appearing multiple times
- "Tool Selector initialized with 5 tools" appearing many times  
- "Warm pool initialized with 5 tool pools" appearing repeatedly
- "[BayesianMemoryPrior] Semantic tool matcher initialized" appearing repeatedly
- "StochasticCostModel initialized" appearing repeatedly

CRITICAL: Query routing degrades from 469ms to 152,048ms (324x slower) 
because components are re-instantiated per-query instead of using singletons.
Each instantiation loads ~300MB SentenceTransformer models that accumulate 
without garbage collection.
"""

import logging
import threading
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Global lock for singleton creation
_lock = threading.Lock()
_instances: dict = {}

# Module-level singletons with individual locks for finer-grained concurrency
_tool_selector: Optional[Any] = None
_reasoning_integration: Optional[Any] = None
_portfolio_executor: Optional[Any] = None
_bayesian_prior: Optional[Any] = None
_warm_pool: Optional[Any] = None
_cost_model: Optional[Any] = None
_semantic_matcher: Optional[Any] = None
_singleton_lock = threading.Lock()


def get_or_create(key: str, factory: Callable) -> Any:
    """
    Thread-safe singleton factory for arbitrary components.
    
    Args:
        key: Unique key for the singleton
        factory: Callable that creates the instance
        
    Returns:
        The singleton instance
    """
    if key not in _instances:
        with _lock:
            if key not in _instances:
                _instances[key] = factory()
    return _instances[key]


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
    global _bayesian_prior, _warm_pool, _cost_model, _semantic_matcher
    
    with _singleton_lock:
        _tool_selector = None
        _reasoning_integration = None
        _portfolio_executor = None
        _bayesian_prior = None
        _warm_pool = None
        _cost_model = None
        _semantic_matcher = None
        _instances.clear()
        logger.info("[Singletons] All singletons reset")


def get_bayesian_prior():
    """
    Get singleton BayesianMemoryPrior instance.
    
    Returns:
        BayesianMemoryPrior instance (singleton).
    """
    global _bayesian_prior
    
    if _bayesian_prior is not None:
        logger.debug("[Singletons] Returning cached BayesianMemoryPrior")
        return _bayesian_prior
    
    with _singleton_lock:
        if _bayesian_prior is not None:
            return _bayesian_prior
        
        logger.info("[Singletons] Creating global BayesianMemoryPrior (ONCE)")
        try:
            from vulcan.reasoning.selection.memory_prior import BayesianMemoryPrior
            _bayesian_prior = BayesianMemoryPrior()
            logger.info("[Singletons] ✓ BayesianMemoryPrior created and cached")
            return _bayesian_prior
        except ImportError as e:
            logger.warning(f"[Singletons] BayesianMemoryPrior not available: {e}")
            return None
        except Exception as e:
            logger.error(f"[Singletons] Failed to create BayesianMemoryPrior: {e}")
            return None


def get_warm_pool():
    """
    Get singleton WarmStartPool instance.
    
    Returns:
        WarmStartPool instance (singleton).
    """
    global _warm_pool
    
    if _warm_pool is not None:
        logger.debug("[Singletons] Returning cached WarmStartPool")
        return _warm_pool
    
    with _singleton_lock:
        if _warm_pool is not None:
            return _warm_pool
        
        logger.info("[Singletons] Creating global WarmStartPool (ONCE)")
        try:
            from vulcan.reasoning.selection.warm_pool import WarmStartPool
            _warm_pool = WarmStartPool()
            logger.info("[Singletons] ✓ WarmStartPool created and cached")
            return _warm_pool
        except ImportError as e:
            logger.warning(f"[Singletons] WarmStartPool not available: {e}")
            return None
        except Exception as e:
            logger.error(f"[Singletons] Failed to create WarmStartPool: {e}")
            return None


def get_cost_model():
    """
    Get singleton StochasticCostModel instance.
    
    Returns:
        StochasticCostModel instance (singleton).
    """
    global _cost_model
    
    if _cost_model is not None:
        logger.debug("[Singletons] Returning cached StochasticCostModel")
        return _cost_model
    
    with _singleton_lock:
        if _cost_model is not None:
            return _cost_model
        
        logger.info("[Singletons] Creating global StochasticCostModel (ONCE)")
        try:
            from vulcan.reasoning.selection.tool_selector import StochasticCostModel
            _cost_model = StochasticCostModel()
            logger.info("[Singletons] ✓ StochasticCostModel created and cached")
            return _cost_model
        except ImportError as e:
            logger.warning(f"[Singletons] StochasticCostModel not available: {e}")
            return None
        except Exception as e:
            logger.error(f"[Singletons] Failed to create StochasticCostModel: {e}")
            return None


def get_semantic_matcher():
    """
    Get singleton SemanticToolMatcher instance.
    
    Returns:
        SemanticToolMatcher instance (singleton).
    """
    global _semantic_matcher
    
    if _semantic_matcher is not None:
        logger.debug("[Singletons] Returning cached SemanticToolMatcher")
        return _semantic_matcher
    
    with _singleton_lock:
        if _semantic_matcher is not None:
            return _semantic_matcher
        
        logger.info("[Singletons] Creating global SemanticToolMatcher (ONCE)")
        try:
            from vulcan.reasoning.selection.semantic_tool_matcher import SemanticToolMatcher
            _semantic_matcher = SemanticToolMatcher()
            logger.info("[Singletons] ✓ SemanticToolMatcher created and cached")
            return _semantic_matcher
        except ImportError as e:
            logger.warning(f"[Singletons] SemanticToolMatcher not available: {e}")
            return None
        except Exception as e:
            logger.error(f"[Singletons] Failed to create SemanticToolMatcher: {e}")
            return None


def prewarm_all():
    """
    Pre-initialize all singletons at startup.
    Call from main.py or unified_platform.py during startup.
    
    This prevents the first query from triggering expensive model loading.
    """
    logger.info("[Singletons] Pre-warming all reasoning singletons...")
    
    results = initialize_all()
    
    # Also initialize the additional components
    bp = get_bayesian_prior()
    results['bayesian_prior'] = bp is not None
    
    wp = get_warm_pool()
    results['warm_pool'] = wp is not None
    
    cm = get_cost_model()
    results['cost_model'] = cm is not None
    
    sm = get_semantic_matcher()
    results['semantic_matcher'] = sm is not None
    
    success_count = sum(1 for v in results.values() if v)
    logger.info(f"[Singletons] ✓ All singletons pre-warmed ({success_count}/{len(results)} initialized)")
    
    return results


def cleanup():
    """
    Release all singletons. Call on shutdown.
    
    This clears all cached singletons to free memory.
    """
    reset_all()
    logger.info("[Singletons] All singletons cleaned up")
