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
_problem_decomposer: Optional[Any] = None
_semantic_bridge: Optional[Any] = None
_world_model: Optional[Any] = None
_self_improvement_drive: Optional[Any] = None
_unified_runtime: Optional[Any] = None
_ai_runtime: Optional[Any] = None
_multimodal_engine: Optional[Any] = None
_singleton_lock = threading.Lock()


def get_or_create(key: str, factory: Callable) -> Any:
    """
    Thread-safe singleton factory for arbitrary components.
    
    Args:
        key: Unique key for the singleton
        factory: Callable that creates the instance
        
    Returns:
        The singleton instance, or None if factory fails
    """
    if key not in _instances:
        with _lock:
            if key not in _instances:
                try:
                    _instances[key] = factory()
                except Exception as e:
                    logger.error(f"[Singletons] Factory failed for '{key}': {e}")
                    return None
    return _instances.get(key)


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
    global _problem_decomposer
    global _semantic_bridge
    global _curiosity_engine
    global _world_model, _self_improvement_drive, _unified_runtime
    global _ai_runtime, _multimodal_engine
    
    with _singleton_lock:
        _tool_selector = None
        _reasoning_integration = None
        _portfolio_executor = None
        _bayesian_prior = None
        _warm_pool = None
        _cost_model = None
        _semantic_matcher = None
        _problem_decomposer = None
        _semantic_bridge = None
        _curiosity_engine = None
        _world_model = None
        _self_improvement_drive = None
        _unified_runtime = None
        _ai_runtime = None
        _multimodal_engine = None
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


def get_warm_pool(tools: Optional[dict] = None, config: Optional[dict] = None):
    """
    Get singleton WarmStartPool instance.
    
    BUG FIX Issues #3, #44: The WarmStartPool was being initialized multiple times
    per query ("Warm pool initialized with 5 tool pools" appearing repeatedly).
    This singleton ensures the pool is created once at startup.
    
    Args:
        tools: Optional dictionary of tool_name -> tool_instance/factory.
               Required on first call if pool needs initialization.
               If None and pool doesn't exist, returns None (safe for early calls).
        config: Optional configuration dictionary for WarmStartPool.
        
    Returns:
        WarmStartPool instance (singleton), or None if tools not provided and pool doesn't exist.
    """
    global _warm_pool
    
    if _warm_pool is not None:
        logger.debug("[Singletons] Returning cached WarmStartPool")
        return _warm_pool
    
    with _singleton_lock:
        if _warm_pool is not None:
            return _warm_pool
        
        # If no tools provided, cannot create WarmStartPool
        # This is safe - callers that need tools will provide them
        if tools is None:
            logger.debug("[Singletons] WarmStartPool not yet initialized (no tools provided)")
            return None
        
        logger.info("[Singletons] Creating global WarmStartPool (ONCE)")
        try:
            from vulcan.reasoning.selection.warm_pool import WarmStartPool
            _warm_pool = WarmStartPool(tools=tools, config=config or {})
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


def get_problem_decomposer():
    """
    Get singleton ProblemDecomposer instance.
    
    The ProblemDecomposer handles hierarchical problem breakdown for complex
    queries (complexity >= 0.40). It uses multiple strategies:
    - ExactDecomposition: Fast pattern matching
    - SemanticDecomposition: Semantic-based decomposition
    - StructuralDecomposition: Structural analysis
    - AnalogicalDecomposition: Analogy-based
    - SyntheticBridging: Synthetic bridging
    - BruteForceSearch: Exhaustive search (last resort)
    
    Returns:
        ProblemDecomposer instance (singleton).
    """
    global _problem_decomposer
    
    if _problem_decomposer is not None:
        logger.debug("[Singletons] Returning cached ProblemDecomposer")
        return _problem_decomposer
    
    with _singleton_lock:
        if _problem_decomposer is not None:
            return _problem_decomposer
        
        logger.info("[Singletons] Creating global ProblemDecomposer (ONCE)")
        try:
            from vulcan.problem_decomposer.decomposer_bootstrap import create_decomposer
            _problem_decomposer = create_decomposer()
            logger.info("[Singletons] ✓ ProblemDecomposer created and cached")
            return _problem_decomposer
        except ImportError as e:
            logger.warning(f"[Singletons] ProblemDecomposer not available: {e}")
            return None
        except Exception as e:
            logger.error(f"[Singletons] Failed to create ProblemDecomposer: {e}")
            return None


def get_semantic_bridge():
    """
    Get singleton SemanticBridge instance.
    
    The SemanticBridge handles cross-domain knowledge transfer, enabling:
    - Concept mapping across domains
    - Transfer validation between domains
    - Conflict resolution when domains disagree
    - Learning from successful cross-domain transfers
    
    Returns:
        SemanticBridge instance (singleton).
    """
    global _semantic_bridge
    
    if _semantic_bridge is not None:
        logger.debug("[Singletons] Returning cached SemanticBridge")
        return _semantic_bridge
    
    with _singleton_lock:
        if _semantic_bridge is not None:
            return _semantic_bridge
        
        logger.info("[Singletons] Creating global SemanticBridge (ONCE)")
        try:
            from vulcan.semantic_bridge import create_semantic_bridge
            _semantic_bridge = create_semantic_bridge(
                world_model=None,  # Will be injected if available
                vulcan_memory=None,
                config={
                    'safety': {'max_risk_score': 0.8, 'require_validation': True},
                    'transfer': {'full_transfer_threshold': 0.8, 'partial_transfer_threshold': 0.5},
                }
            )
            logger.info("[Singletons] ✓ SemanticBridge created and cached")
            return _semantic_bridge
        except ImportError as e:
            logger.warning(f"[Singletons] SemanticBridge not available: {e}")
            return None
        except Exception as e:
            logger.error(f"[Singletons] Failed to create SemanticBridge: {e}")
            return None


def prewarm_all():
    """
    Pre-initialize all singletons at startup.
    Call from main.py or unified_platform.py during startup.
    
    This prevents the first query from triggering expensive model loading.
    
    BUG FIX Issues #1-5, #27-28: Now includes WorldModel, SelfImprovementDrive,
    UnifiedRuntime, AIRuntime, and MultiModalReasoningEngine to prevent 
    per-query reinitialization.
    """
    logger.info("[Singletons] Pre-warming all singletons...")
    
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
    
    pd = get_problem_decomposer()
    results['problem_decomposer'] = pd is not None
    
    sb = get_semantic_bridge()
    results['semantic_bridge'] = sb is not None
    
    ce = get_curiosity_engine()
    results['curiosity_engine'] = ce is not None
    
    # BUG FIX Issues #1-5, #27: Pre-warm critical per-query reinitialized components
    wm = get_world_model()
    results['world_model'] = wm is not None
    
    sid = get_self_improvement_drive()
    results['self_improvement_drive'] = sid is not None
    
    ur = get_unified_runtime()
    results['unified_runtime'] = ur is not None
    
    # BUG FIX Issues #2-3, #28: Pre-warm AIRuntime and MultiModalReasoningEngine
    ar = get_ai_runtime()
    results['ai_runtime'] = ar is not None
    
    mme = get_multimodal_engine()
    results['multimodal_engine'] = mme is not None
    
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


# ============================================
# CURIOSITY ENGINE SINGLETON
# ============================================

_curiosity_engine: Optional[Any] = None
_curiosity_engine_lock = threading.Lock()


def get_curiosity_engine() -> Optional[Any]:
    """
    Get singleton CuriosityEngine instance.
    
    The CuriosityEngine is the main orchestrator for curiosity-driven learning,
    including knowledge gap detection, experiment generation, and exploration
    budget management.
    
    Returns:
        CuriosityEngine instance (singleton), or None if unavailable.
        
    Note:
        Returns None if CuriosityEngine is not available (e.g., missing numpy).
        Always check the return value before using.
    """
    global _curiosity_engine
    
    # Check availability flag first
    try:
        from vulcan.curiosity_engine import CURIOSITY_ENGINE_AVAILABLE
        if not CURIOSITY_ENGINE_AVAILABLE:
            logger.debug("[Singletons] CuriosityEngine not available (dependencies missing)")
            return None
    except ImportError:
        logger.debug("[Singletons] curiosity_engine module not importable")
        return None
    
    if _curiosity_engine is not None:
        logger.debug("[Singletons] Returning cached CuriosityEngine")
        return _curiosity_engine
    
    with _curiosity_engine_lock:
        if _curiosity_engine is not None:
            return _curiosity_engine
        
        logger.info("[Singletons] Creating global CuriosityEngine (ONCE)")
        try:
            from vulcan.curiosity_engine import CuriosityEngine, get_curiosity_engine as ce_get
            
            # Try the factory function first (it may return a pre-existing instance)
            if ce_get is not None:
                engine = ce_get()
                if engine is not None:
                    _curiosity_engine = engine
                    logger.info("[Singletons] ✓ CuriosityEngine obtained from factory")
                    return _curiosity_engine
            
            # Otherwise create new instance
            if CuriosityEngine is not None:
                _curiosity_engine = CuriosityEngine()
                logger.info("[Singletons] ✓ CuriosityEngine created and cached")
                return _curiosity_engine
            
            logger.warning("[Singletons] CuriosityEngine class is None")
            return None
            
        except ImportError as e:
            logger.warning(f"[Singletons] CuriosityEngine import failed: {e}")
            return None
        except Exception as e:
            logger.error(f"[Singletons] Failed to create CuriosityEngine: {e}")
            return None


# ============================================
# WORLD MODEL SINGLETON (Issue #1-4)
# ============================================

_world_model_lock = threading.Lock()


def get_world_model(config: Optional[dict] = None) -> Optional[Any]:
    """
    Get singleton WorldModel instance.
    
    BUG FIX Issue #1-4: The WorldModel was being reinitialized per-query,
    causing ~10-15 seconds of initialization overhead. This singleton ensures
    it's only initialized once at startup.
    
    Args:
        config: Optional config dict for first-time initialization
        
    Returns:
        WorldModel instance (singleton), or None if unavailable.
    """
    global _world_model
    
    if _world_model is not None:
        logger.debug("[Singletons] Returning cached WorldModel")
        return _world_model
    
    with _world_model_lock:
        if _world_model is not None:
            return _world_model
        
        logger.info("[Singletons] Creating global WorldModel (ONCE)")
        try:
            from vulcan.world_model.world_model_core import WorldModel
            _world_model = WorldModel(config=config)
            logger.info("[Singletons] ✓ WorldModel created and cached")
            return _world_model
        except ImportError as e:
            logger.warning(f"[Singletons] WorldModel import failed: {e}")
            return None
        except Exception as e:
            logger.error(f"[Singletons] Failed to create WorldModel: {e}")
            return None


# ============================================
# SELF-IMPROVEMENT DRIVE SINGLETON (Issue #5)
# ============================================

_self_improvement_lock = threading.Lock()


def get_self_improvement_drive(config_path: Optional[str] = None, state_path: Optional[str] = None) -> Optional[Any]:
    """
    Get singleton SelfImprovementDrive instance.
    
    BUG FIX Issue #5: The SelfImprovementDrive was reloading state file repeatedly.
    This singleton ensures state is loaded once and persisted in memory.
    
    Args:
        config_path: Optional path to config file (first-time only)
        state_path: Optional path to state file (first-time only)
        
    Returns:
        SelfImprovementDrive instance (singleton), or None if unavailable.
    """
    global _self_improvement_drive
    
    if _self_improvement_drive is not None:
        logger.debug("[Singletons] Returning cached SelfImprovementDrive")
        return _self_improvement_drive
    
    with _self_improvement_lock:
        if _self_improvement_drive is not None:
            return _self_improvement_drive
        
        logger.info("[Singletons] Creating global SelfImprovementDrive (ONCE)")
        try:
            from vulcan.world_model.meta_reasoning.self_improvement_drive import SelfImprovementDrive
            _self_improvement_drive = SelfImprovementDrive(
                config_path=config_path,
                state_path=state_path
            )
            logger.info("[Singletons] ✓ SelfImprovementDrive created and cached")
            return _self_improvement_drive
        except ImportError as e:
            logger.warning(f"[Singletons] SelfImprovementDrive import failed: {e}")
            return None
        except Exception as e:
            logger.error(f"[Singletons] Failed to create SelfImprovementDrive: {e}")
            return None


# ============================================
# UNIFIED RUNTIME SINGLETON (Issue #27)
# ============================================

_unified_runtime_lock = threading.Lock()


def get_unified_runtime(config: Optional[Any] = None) -> Optional[Any]:
    """
    Get singleton UnifiedRuntime instance.
    
    BUG FIX Issue #27: The UnifiedRuntime was loading manifest file on every query.
    This singleton ensures manifest is loaded once at startup.
    
    Args:
        config: Optional config for first-time initialization
        
    Returns:
        UnifiedRuntime instance (singleton), or None if unavailable.
    """
    global _unified_runtime
    
    if _unified_runtime is not None:
        logger.debug("[Singletons] Returning cached UnifiedRuntime")
        return _unified_runtime
    
    with _unified_runtime_lock:
        if _unified_runtime is not None:
            return _unified_runtime
        
        logger.info("[Singletons] Creating global UnifiedRuntime (ONCE)")
        try:
            from unified_runtime.unified_runtime_core import UnifiedRuntime
            _unified_runtime = UnifiedRuntime(config=config)
            logger.info("[Singletons] ✓ UnifiedRuntime created and cached")
            return _unified_runtime
        except ImportError as e:
            logger.warning(f"[Singletons] UnifiedRuntime import failed: {e}")
            return None
        except Exception as e:
            logger.error(f"[Singletons] Failed to create UnifiedRuntime: {e}")
            return None


# ============================================
# AI RUNTIME SINGLETON (Issue #28)
# ============================================

_ai_runtime_lock = threading.Lock()


def get_ai_runtime(config: Optional[dict] = None) -> Optional[Any]:
    """
    Get singleton AIRuntime instance.
    
    BUG FIX Issue #28: The AIRuntime was registering OpenAI provider multiple times
    because it was being re-instantiated per-query. This singleton ensures
    providers are registered only once at startup.
    
    Args:
        config: Optional config dict for first-time initialization
        
    Returns:
        AIRuntime instance (singleton), or None if unavailable.
    """
    global _ai_runtime
    
    if _ai_runtime is not None:
        logger.debug("[Singletons] Returning cached AIRuntime")
        return _ai_runtime
    
    with _ai_runtime_lock:
        if _ai_runtime is not None:
            return _ai_runtime
        
        logger.info("[Singletons] Creating global AIRuntime (ONCE)")
        try:
            from unified_runtime.ai_runtime_integration import AIRuntime
            _ai_runtime = AIRuntime(config=config)
            logger.info("[Singletons] ✓ AIRuntime created and cached")
            return _ai_runtime
        except ImportError as e:
            logger.warning(f"[Singletons] AIRuntime import failed: {e}")
            return None
        except Exception as e:
            logger.error(f"[Singletons] Failed to create AIRuntime: {e}")
            return None


# ============================================
# MULTIMODAL REASONING ENGINE SINGLETON (Issue #2-3)
# ============================================

_multimodal_engine_lock = threading.Lock()


def get_multimodal_engine(enable_learning: bool = True, device: str = "cpu") -> Optional[Any]:
    """
    Get singleton MultiModalReasoningEngine instance.
    
    BUG FIX Issues #2-3: The MultiModalReasoningEngine was logging
    "Neural reasoning modules initialized successfully" multiple times
    per query because it was being re-instantiated. This singleton ensures
    neural modules are initialized only once at startup.
    
    Args:
        enable_learning: Whether to enable learning mode (first-time only)
        device: Device to use - "cpu" or "cuda" (first-time only)
        
    Returns:
        MultiModalReasoningEngine instance (singleton), or None if unavailable.
    """
    global _multimodal_engine
    
    if _multimodal_engine is not None:
        logger.debug("[Singletons] Returning cached MultiModalReasoningEngine")
        return _multimodal_engine
    
    with _multimodal_engine_lock:
        if _multimodal_engine is not None:
            return _multimodal_engine
        
        logger.info("[Singletons] Creating global MultiModalReasoningEngine (ONCE)")
        try:
            from vulcan.reasoning.multimodal_reasoning import MultiModalReasoningEngine
            _multimodal_engine = MultiModalReasoningEngine(
                enable_learning=enable_learning,
                device=device
            )
            logger.info("[Singletons] ✓ MultiModalReasoningEngine created and cached")
            return _multimodal_engine
        except ImportError as e:
            logger.warning(f"[Singletons] MultiModalReasoningEngine import failed: {e}")
            return None
        except Exception as e:
            logger.error(f"[Singletons] Failed to create MultiModalReasoningEngine: {e}")
            return None
