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
import os
import threading
from pathlib import Path
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
_unified_reasoner: Optional[Any] = None  # Note: Add singleton for UnifiedReasoner
_math_verification_engine: Optional[Any] = None  # CACHING FIX: Singleton for MathematicalVerificationEngine
_hierarchical_memory: Optional[Any] = None  # PERF FIX Issue #2: Singleton for HierarchicalMemory
_unified_learning_system: Optional[Any] = None  # PERF FIX Issue #5: Singleton for UnifiedLearningSystem
_singleton_lock = threading.Lock()


def _get_tool_selector_state_path() -> Path:
    """Get the path for tool selector state persistence."""
    # Use environment variable or default to .vulcan_state directory
    state_dir = os.environ.get("VULCAN_STATE_DIR", ".vulcan_state")
    return Path(state_dir) / "tool_selector"


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
    
    Note: Tool Weight Memory Amnesia - Now loads persisted state at startup
    so bandit weights survive restarts.
    
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
            
            # Note: Tool Weight Memory Amnesia - Load persisted state
            state_path = _get_tool_selector_state_path()
            if state_path.exists():
                try:
                    _tool_selector.load_state(str(state_path))
                    logger.info(f"[Singletons] ✓ ToolSelector state loaded from {state_path}")
                except Exception as load_err:
                    logger.warning(f"[Singletons] Could not load ToolSelector state: {load_err}")
            
            logger.info("[Singletons] ✓ ToolSelector created and cached")
            return _tool_selector
        except ImportError as e:
            logger.warning(f"[Singletons] ToolSelector not available: {e}")
            return None
        except Exception as e:
            logger.error(f"[Singletons] Failed to create ToolSelector: {e}")
            return None


def save_tool_selector_state():
    """
    Note: Tool Weight Memory Amnesia - Save ToolSelector state to disk.
    
    Call this periodically or on shutdown to persist bandit weights and other
    learned state so it survives restarts.
    
    Returns:
        bool: True if save succeeded, False otherwise.
    """
    global _tool_selector
    
    if _tool_selector is None:
        logger.debug("[Singletons] No ToolSelector to save")
        return False
    
    try:
        state_path = _get_tool_selector_state_path()
        # Ensure parent directory exists
        state_path.parent.mkdir(parents=True, exist_ok=True)
        _tool_selector.save_state(str(state_path))
        logger.info(f"[Singletons] ✓ ToolSelector state saved to {state_path}")
        return True
    except Exception as e:
        logger.error(f"[Singletons] Failed to save ToolSelector state: {e}")
        return False


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
            from vulcan.reasoning.integration import ReasoningIntegration
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


# ============================================
# UNIFIED REASONER SINGLETON
# ============================================

_unified_reasoner_lock = threading.Lock()


def get_unified_reasoner(
    enable_learning: bool = True,
    enable_safety: bool = True,
    config: Optional[dict] = None
) -> Optional[Any]:
    """
    Get singleton UnifiedReasoner instance.
    
    Note: The UnifiedReasoner was being instantiated per-query, causing
    WarmStartPool, ToolSelector, and MultiModalReasoningEngine to be
    reinitialized each time. This singleton ensures unified reasoning
    components are created once at startup.
    
    Args:
        enable_learning: Whether to enable learning (first-time only)
        enable_safety: Whether to enable safety checks (first-time only)
        config: Optional config dict (first-time only)
        
    Returns:
        UnifiedReasoner instance (singleton), or None if unavailable.
    """
    global _unified_reasoner
    
    if _unified_reasoner is not None:
        logger.debug("[Singletons] Returning cached UnifiedReasoner")
        return _unified_reasoner
    
    with _unified_reasoner_lock:
        if _unified_reasoner is not None:
            return _unified_reasoner
        
        logger.info("[Singletons] Creating global UnifiedReasoner (ONCE)")
        try:
            from vulcan.reasoning.unified import UnifiedReasoner
            _unified_reasoner = UnifiedReasoner(
                enable_learning=enable_learning,
                enable_safety=enable_safety,
                config=config
            )
            logger.info("[Singletons] ✓ UnifiedReasoner created and cached")
            return _unified_reasoner
        except ImportError as e:
            logger.warning(f"[Singletons] UnifiedReasoner import failed: {e}")
            return None
        except Exception as e:
            logger.error(f"[Singletons] Failed to create UnifiedReasoner: {e}")
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
    global _ai_runtime, _multimodal_engine, _unified_reasoner
    global _math_verification_engine, _hierarchical_memory
    global _unified_learning_system
    
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
        _unified_reasoner = None
        _math_verification_engine = None
        _hierarchical_memory = None
        _unified_learning_system = None
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
    
    Note: The WarmStartPool was being initialized multiple times
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
    
    Note: Now includes WorldModel, SelfImprovementDrive,
    UnifiedRuntime, AIRuntime, and MultiModalReasoningEngine to prevent 
    per-query reinitialization.
    
    CACHING FIX: Now includes MathematicalVerificationEngine to prevent
    "MathematicalVerificationEngine initialized" appearing 4+ times per request.
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
    
    # Note Issues #1-5, #27: Pre-warm critical per-query reinitialized components
    wm = get_world_model()
    results['world_model'] = wm is not None
    
    sid = get_self_improvement_drive()
    results['self_improvement_drive'] = sid is not None
    
    ur = get_unified_runtime()
    results['unified_runtime'] = ur is not None
    
    # Note Issues #2-3, #28: Pre-warm AIRuntime and MultiModalReasoningEngine
    ar = get_ai_runtime()
    results['ai_runtime'] = ar is not None
    
    mme = get_multimodal_engine()
    results['multimodal_engine'] = mme is not None
    
    # CACHING FIX: Pre-warm MathematicalVerificationEngine
    mve = get_math_verification_engine()
    results['math_verification_engine'] = mve is not None
    
    # PERF FIX Issue #5: Pre-warm UnifiedLearningSystem to persist ensemble weights
    uls = get_unified_learning_system()
    results['unified_learning_system'] = uls is not None
    
    success_count = sum(1 for v in results.values() if v)
    logger.info(f"[Singletons] ✓ All singletons pre-warmed ({success_count}/{len(results)} initialized)")
    
    return results


def cleanup():
    """
    Release all singletons. Call on shutdown.
    
    Note: Tool Weight Memory Amnesia - Now saves ToolSelector state before cleanup
    so bandit weights persist across restarts.
    
    This clears all cached singletons to free memory.
    """
    # Note: Save tool selector state before cleanup
    save_tool_selector_state()
    
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
    
    Note: The WorldModel was being reinitialized per-query,
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
    
    Note: The SelfImprovementDrive was reloading state file repeatedly.
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
    
    Note: The UnifiedRuntime was loading manifest file on every query.
    This singleton ensures manifest is loaded once at startup.
    
    Note: Prevent repeated initialization/shutdown cycles by ensuring
    only one UnifiedRuntime instance exists across the entire application.
    Multiple fallback patterns in graphix_arena.py, main.py, and deployment.py
    were creating separate instances when singleton returned None.
    
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
            # Note Issue #1: Mark as singleton so __del__ knows to cleanup
            _unified_runtime._is_singleton = True
            UnifiedRuntime._singleton_instance = _unified_runtime
            logger.info("[Singletons] ✓ UnifiedRuntime created and cached")
            return _unified_runtime
        except ImportError as e:
            logger.warning(f"[Singletons] UnifiedRuntime import failed: {e}")
            return None
        except Exception as e:
            logger.error(f"[Singletons] Failed to create UnifiedRuntime: {e}")
            return None


def set_unified_runtime(runtime: Any) -> bool:
    """
    Note: Register a UnifiedRuntime instance with the singleton registry.
    
    This allows callers to register a manually-created UnifiedRuntime instance
    with the singleton registry, preventing duplicate instances when fallback
    patterns create their own.
    
    Args:
        runtime: The UnifiedRuntime instance to register as the singleton
        
    Returns:
        True if registered successfully, False if singleton already exists
    """
    global _unified_runtime
    
    with _unified_runtime_lock:
        if _unified_runtime is None:
            _unified_runtime = runtime
            # Note Issue #1: Mark as singleton so __del__ knows to cleanup
            # Note: _is_singleton is always initialized in UnifiedRuntime.__init__()
            runtime._is_singleton = True
            logger.info("[Singletons] UnifiedRuntime registered from external source")
            return True
        elif _unified_runtime is not runtime:
            logger.debug(
                "[Singletons] UnifiedRuntime already exists - ignoring duplicate"
            )
            return False
        return True  # Already registered (same instance)


def get_or_create_unified_runtime(config: Optional[Any] = None) -> Optional[Any]:
    """
    Note: Get or create UnifiedRuntime with automatic singleton registration.
    
    This is a convenience function that:
    1. Tries to get the singleton UnifiedRuntime
    2. If singleton returns None, creates a new instance directly
    3. Registers the new instance with the singleton to prevent duplicates
    
    Use this instead of the fallback pattern:
        runtime = get_unified_runtime()
        if runtime is None:
            runtime = UnifiedRuntime()
            set_unified_runtime(runtime)
    
    Args:
        config: Optional config for first-time initialization
        
    Returns:
        UnifiedRuntime instance (singleton or newly created), or None if import fails
    """
    global _unified_runtime
    
    # First try the singleton
    runtime = get_unified_runtime(config)
    if runtime is not None:
        return runtime
    
    # Singleton failed - try direct instantiation and register
    with _unified_runtime_lock:
        # Double-check in case another thread created it
        if _unified_runtime is not None:
            return _unified_runtime
        
        try:
            from unified_runtime.unified_runtime_core import UnifiedRuntime
            _unified_runtime = UnifiedRuntime(config=config)
            # Note Issue #1: Mark as singleton so __del__ knows to cleanup
            _unified_runtime._is_singleton = True
            UnifiedRuntime._singleton_instance = _unified_runtime
            logger.info("[Singletons] ✓ UnifiedRuntime created via fallback and cached")
            return _unified_runtime
        except ImportError as e:
            logger.warning(f"[Singletons] UnifiedRuntime import failed in fallback: {e}")
            return None
        except Exception as e:
            logger.error(f"[Singletons] Failed to create UnifiedRuntime in fallback: {e}")
            return None


# ============================================
# AI RUNTIME SINGLETON (Issue #28)
# ============================================

_ai_runtime_lock = threading.Lock()


def get_ai_runtime(config: Optional[dict] = None) -> Optional[Any]:
    """
    Get singleton AIRuntime instance.
    
    Note: The AIRuntime was registering OpenAI provider multiple times
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
    
    Note: The MultiModalReasoningEngine was logging
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


# ============================================
# MATHEMATICAL VERIFICATION ENGINE SINGLETON (CACHING FIX)
# ============================================

_math_verification_engine_lock = threading.Lock()


def get_math_verification_engine() -> Optional[Any]:
    """
    Get singleton MathematicalVerificationEngine instance.
    
    CACHING FIX: The MathematicalVerificationEngine was being re-initialized
    4+ times per query ("MathematicalVerificationEngine initialized" appearing
    repeatedly in logs). This singleton ensures the engine is created once
    at startup.
    
    Returns:
        MathematicalVerificationEngine instance (singleton), or None if unavailable.
    """
    global _math_verification_engine
    
    if _math_verification_engine is not None:
        logger.debug("[Singletons] Returning cached MathematicalVerificationEngine")
        return _math_verification_engine
    
    with _math_verification_engine_lock:
        if _math_verification_engine is not None:
            return _math_verification_engine
        
        logger.info("[Singletons] Creating global MathematicalVerificationEngine (ONCE)")
        try:
            from vulcan.reasoning.mathematical_verification import MathematicalVerificationEngine
            _math_verification_engine = MathematicalVerificationEngine()
            logger.info("[Singletons] ✓ MathematicalVerificationEngine created and cached")
            return _math_verification_engine
        except ImportError as e:
            logger.warning(f"[Singletons] MathematicalVerificationEngine import failed: {e}")
            return None
        except Exception as e:
            logger.error(f"[Singletons] Failed to create MathematicalVerificationEngine: {e}")
            return None


# ============================================
# HIERARCHICAL MEMORY SINGLETON (Issue #2)
# ============================================

_hierarchical_memory_lock = threading.Lock()


def get_hierarchical_memory(config: Optional[Any] = None) -> Optional[Any]:
    """
    Get singleton HierarchicalMemory instance.
    
    PERF FIX Issue #2: The HierarchicalMemory was being re-instantiated in
    multiple places (GraphixVulcanLLM, agent_pool, etc.), causing:
    - "[HierarchicalMemory] Using model from global registry" appearing per request
    - Repeated embedding model loading/initialization overhead
    - Memory accumulation from multiple instances
    
    This singleton ensures HierarchicalMemory is created once at startup and
    shared across all components.
    
    Args:
        config: Optional MemoryConfig for first-time initialization.
               If None, uses default MemoryConfig.
        
    Returns:
        HierarchicalMemory instance (singleton), or None if unavailable.
    """
    global _hierarchical_memory
    
    if _hierarchical_memory is not None:
        logger.debug("[Singletons] Returning cached HierarchicalMemory")
        return _hierarchical_memory
    
    with _hierarchical_memory_lock:
        if _hierarchical_memory is not None:
            return _hierarchical_memory
        
        logger.info("[Singletons] Creating global HierarchicalMemory (ONCE)")
        try:
            from vulcan.memory.hierarchical import HierarchicalMemory
            from vulcan.memory.base import MemoryConfig
            
            # Use provided config or create default
            if config is None:
                config = MemoryConfig(max_working_memory=50)
            
            _hierarchical_memory = HierarchicalMemory(config)
            logger.info("[Singletons] ✓ HierarchicalMemory created and cached")
            return _hierarchical_memory
        except ImportError as e:
            logger.warning(f"[Singletons] HierarchicalMemory import failed: {e}")
            return None
        except Exception as e:
            logger.error(f"[Singletons] Failed to create HierarchicalMemory: {e}")
            return None


# ============================================
# UNIFIED LEARNING SYSTEM SINGLETON (Issue #5)
# ============================================

_unified_learning_system_lock = threading.Lock()


def get_unified_learning_system() -> Optional[Any]:
    """
    Get singleton UnifiedLearningSystem instance.
    
    PERF FIX Issue #5: The UnifiedLearningSystem was being re-instantiated
    per request, causing:
    - "[Ensemble] All weights are zero - using uniform weights" every request
    - Tool weight adjustments being lost between requests
    - Learning state not persisting across queries
    - LearningStatePersistence re-loading from disk repeatedly
    
    With the singleton pattern, the learning system:
    - Initializes once at startup with LearningStatePersistence
    - Maintains tool weight adjustments across all requests
    - Ensemble weights persist and accumulate properly
    - Learning feedback actually improves routing over time
    
    Returns:
        UnifiedLearningSystem instance (singleton), or None if unavailable.
    """
    global _unified_learning_system
    
    if _unified_learning_system is not None:
        logger.debug("[Singletons] Returning cached UnifiedLearningSystem")
        return _unified_learning_system
    
    with _unified_learning_system_lock:
        if _unified_learning_system is not None:
            return _unified_learning_system
        
        logger.info("[Singletons] Creating global UnifiedLearningSystem (ONCE)")
        try:
            from vulcan.learning import UnifiedLearningSystem
            
            _unified_learning_system = UnifiedLearningSystem()
            logger.info("[Singletons] ✓ UnifiedLearningSystem created and cached")
            logger.info("[Singletons] ✓ Tool weights will persist across requests")
            return _unified_learning_system
        except ImportError as e:
            logger.warning(f"[Singletons] UnifiedLearningSystem import failed: {e}")
            return None
        except Exception as e:
            logger.error(f"[Singletons] Failed to create UnifiedLearningSystem: {e}")
            return None


# ============================================
# LLM CLIENT SINGLETON (Task 4 Fix)
# ============================================

_llm_client: Optional[Any] = None
_llm_client_lock = threading.Lock()


def get_llm_client() -> Optional[Any]:
    """
    Get the global LLM client singleton.
    
    FIX TASK 4: Tools were being initialized with llm=None because the LLM
    client wasn't available during component initialization. This singleton
    provides a centralized way to access the LLM client from anywhere.
    
    Tries multiple sources:
    1. Cached singleton
    2. HybridLLMExecutor's local_llm
    3. Global GraphixVulcanLLM instance
    
    Returns:
        LLM client instance, or None if unavailable.
    """
    global _llm_client
    
    if _llm_client is not None:
        return _llm_client
    
    with _llm_client_lock:
        if _llm_client is not None:
            return _llm_client
        
        # Try HybridLLMExecutor first
        try:
            from vulcan.llm import get_hybrid_executor
            hybrid_executor = get_hybrid_executor()
            if hybrid_executor is not None:
                client = getattr(hybrid_executor, 'local_llm', None)
                if client is not None:
                    _llm_client = client
                    logger.info("[Singletons] ✓ LLM client obtained from HybridLLMExecutor")
                    return _llm_client
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"[Singletons] Failed to get LLM from hybrid executor: {e}")
        
        # Try global GraphixVulcanLLM singleton getter if available
        try:
            from vulcan.llm import GraphixVulcanLLM
            # Try to use a public singleton getter if available
            if hasattr(GraphixVulcanLLM, 'get_instance'):
                instance = GraphixVulcanLLM.get_instance()
                if instance is not None:
                    _llm_client = instance
                    logger.info("[Singletons] ✓ LLM client obtained from GraphixVulcanLLM.get_instance()")
                    return _llm_client
            # Fallback: Try to get from module-level singleton registry
            # Note: Accessing internal attributes is not ideal but may be necessary
            # for backward compatibility with existing code patterns
            elif hasattr(GraphixVulcanLLM, 'default_instance'):
                instance = GraphixVulcanLLM.default_instance
                if instance is not None:
                    _llm_client = instance
                    logger.info("[Singletons] ✓ LLM client obtained from GraphixVulcanLLM.default_instance")
                    return _llm_client
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"[Singletons] Failed to get GraphixVulcanLLM: {e}")
        
        logger.debug("[Singletons] LLM client not available yet")
        return None


def set_llm_client(client: Any) -> None:
    """
    Set the global LLM client singleton.
    
    This should be called during app initialization to make the LLM client
    available to all reasoning components.
    
    Args:
        client: The LLM client instance to set as the global singleton.
    """
    global _llm_client
    with _llm_client_lock:
        _llm_client = client
        logger.info("[Singletons] ✓ LLM client registered")
