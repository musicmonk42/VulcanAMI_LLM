"""
Vulcan Reasoning Module

Exports all reasoning components and types for the unified reasoning system.
This module provides access to multiple reasoning paradigms and orchestration.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ============================================================================
# Core Types - Always available
# ============================================================================
try:
    from .reasoning_types import (
        ReasoningType,
        ReasoningStep,
        ReasoningChain,
        ReasoningResult,
        ModalityType,
        SelectionMode,
        PortfolioStrategy,
        UtilityContext,
    )
    TYPES_AVAILABLE = True
except ImportError as e:
    logger.critical(f"Core types import failed: {e}")
    TYPES_AVAILABLE = False
    # Create minimal fallbacks
    from enum import Enum
    class ReasoningType(Enum):
        SYMBOLIC = "symbolic"
        CAUSAL = "causal"
        PROBABILISTIC = "probabilistic"
        ANALOGICAL = "analogical"
        MULTIMODAL = "multimodal"

# ============================================================================
# Strategy Enum - Import from unified_reasoning or provide fallback
# ============================================================================
try:
    from .unified_reasoning import ReasoningStrategy
except ImportError as e:
    logger.warning(f"Could not import ReasoningStrategy from unified_reasoning: {e}")
    from enum import Enum
    
    class ReasoningStrategy(Enum):
        """Fallback ReasoningStrategy enum"""
        SINGLE = "single"
        SEQUENTIAL = "sequential"
        PARALLEL = "parallel"
        ENSEMBLE = "ensemble"
        HIERARCHICAL = "hierarchical"
        ADAPTIVE = "adaptive"
        HYBRID = "hybrid"
        PORTFOLIO = "portfolio"
        UTILITY_BASED = "utility_based"

# ============================================================================
# Probabilistic Reasoning - Core component
# ============================================================================
try:
    from .probabilistic_reasoning import ProbabilisticReasoner
    PROBABILISTIC_AVAILABLE = True
except ImportError as e:
    logger.error(f"Probabilistic reasoning import failed: {e}")
    ProbabilisticReasoner = None
    PROBABILISTIC_AVAILABLE = False

# ============================================================================
# Causal Reasoning - Core component
# ============================================================================
try:
    from .causal_reasoning import (
        CausalReasoner,
        EnhancedCausalReasoning,
        CausalEdge
    )
    CAUSAL_AVAILABLE = True
except ImportError as e:
    logger.error(f"Causal reasoning import failed: {e}")
    CausalReasoner = None
    EnhancedCausalReasoning = None
    CausalEdge = None
    CAUSAL_AVAILABLE = False

# ============================================================================
# Symbolic Reasoning - Core component (FIXED: Ultra-defensive imports)
# ============================================================================
try:
    # Primary import from subdirectory - most critical
    from .symbolic.reasoner import SymbolicReasoner
    SYMBOLIC_AVAILABLE = True
    logger.info("Symbolic reasoning loaded successfully from symbolic.reasoner")
    
    # Try to import core types - but don't fail if they're not there
    # Import each one individually with its own try/except
    try:
        from .symbolic.core import Clause
    except (ImportError, AttributeError) as e:
        logger.debug(f"Could not import Clause: {e}")
        Clause = None
    
    try:
        from .symbolic.core import Literal
    except (ImportError, AttributeError) as e:
        logger.debug(f"Could not import Literal: {e}")
        Literal = None
    
    try:
        from .symbolic.core import Constant
    except (ImportError, AttributeError) as e:
        logger.debug(f"Could not import Constant: {e}")
        Constant = None
    
    try:
        from .symbolic.core import Variable
    except (ImportError, AttributeError) as e:
        logger.debug(f"Could not import Variable: {e}")
        Variable = None
    
    # Try extended symbolic reasoners - optional
    try:
        from .symbolic.reasoner import BayesianReasoner
    except (ImportError, AttributeError) as e:
        logger.debug(f"BayesianReasoner not available: {e}")
        BayesianReasoner = None
    
    try:
        from .symbolic.reasoner import EnhancedSymbolicReasoner
    except (ImportError, AttributeError) as e:
        logger.debug(f"EnhancedSymbolicReasoner not available: {e}")
        EnhancedSymbolicReasoner = None
    
except ImportError as e:
    logger.error(f"Symbolic reasoning import failed: {e}")
    SymbolicReasoner = None
    BayesianReasoner = None
    EnhancedSymbolicReasoner = None
    Clause = None
    Literal = None
    Constant = None
    Variable = None
    SYMBOLIC_AVAILABLE = False
except Exception as e:
    logger.error(f"Unexpected error loading symbolic reasoning: {e}")
    SymbolicReasoner = None
    BayesianReasoner = None
    EnhancedSymbolicReasoner = None
    Clause = None
    Literal = None
    Constant = None
    Variable = None
    SYMBOLIC_AVAILABLE = False

# ============================================================================
# Analogical Reasoning - Core component
# ============================================================================
try:
    from .analogical_reasoning import (
        AnalogicalReasoningEngine,
        AnalogicalReasoner,
        Entity,
        Relation
    )
    ANALOGICAL_AVAILABLE = True
except ImportError as e:
    logger.error(f"Analogical reasoning import failed: {e}")
    AnalogicalReasoningEngine = None
    AnalogicalReasoner = None
    Entity = None
    Relation = None
    ANALOGICAL_AVAILABLE = False

# ============================================================================
# Multimodal Reasoning - Core component
# ============================================================================
try:
    from .multimodal_reasoning import (
        MultiModalReasoningEngine,
        MultimodalReasoner
    )
    MULTIMODAL_AVAILABLE = True
except ImportError as e:
    logger.error(f"Multimodal reasoning import failed: {e}")
    MultiModalReasoningEngine = None
    MultimodalReasoner = None
    MULTIMODAL_AVAILABLE = False

# ============================================================================
# Unified Reasoner - Main orchestrator (critical)
# ============================================================================
try:
    from .unified_reasoning import UnifiedReasoner
    UNIFIED_AVAILABLE = True
except ImportError as e:
    logger.critical(f"UnifiedReasoner import failed: {e}")
    UnifiedReasoner = None
    UNIFIED_AVAILABLE = False

# ============================================================================
# Explainability - Optional enhancement
# ============================================================================
try:
    from .reasoning_explainer import (
        ReasoningExplainer,
        SafetyAwareReasoning
    )
    EXPLAINER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Reasoning explainer import failed: {e}")
    ReasoningExplainer = None
    SafetyAwareReasoning = None
    EXPLAINER_AVAILABLE = False

# ============================================================================
# Tool Selection Components - Optional but recommended
# ============================================================================
try:
    from .selection.tool_selector import (
        ToolSelector,
        SelectionRequest,
        SelectionResult
    )
    from .selection.admission_control import (
        AdmissionControlIntegration,
        RequestPriority
    )
    SELECTION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Selection components import failed: {e}")
    ToolSelector = None
    SelectionRequest = None
    SelectionResult = None
    AdmissionControlIntegration = None
    RequestPriority = None
    SELECTION_AVAILABLE = False

# ============================================================================
# Contextual Bandit - Advanced feature
# ============================================================================
try:
    from .contextual_bandit import (
        AdaptiveBanditOrchestrator,
        BanditContext,
        BanditFeedback,
        BanditAction
    )
    BANDIT_AVAILABLE = True
except ImportError as e:
    logger.debug(f"Contextual bandit import failed: {e}")
    AdaptiveBanditOrchestrator = None
    BanditContext = None
    BanditFeedback = None
    BanditAction = None
    BANDIT_AVAILABLE = False

# ============================================================================
# Public API - What users can import
# ============================================================================
__all__ = [
    # ===== Core Types =====
    'ReasoningType',
    'ReasoningStep',
    'ReasoningChain',
    'ReasoningResult',
    'ModalityType',
    'SelectionMode',
    'PortfolioStrategy',
    'UtilityContext',
    'ReasoningStrategy',
    
    # ===== Main Orchestrator =====
    'UnifiedReasoner',
    
    # ===== Individual Reasoners =====
    'ProbabilisticReasoner',
    'CausalReasoner',
    'EnhancedCausalReasoning',
    'SymbolicReasoner',
    'BayesianReasoner',
    'EnhancedSymbolicReasoner',
    'AnalogicalReasoningEngine',
    'AnalogicalReasoner',
    'MultiModalReasoningEngine',
    'MultimodalReasoner',
    
    # ===== Symbolic Types =====
    'Clause',
    'Literal',
    'Constant',
    'Variable',
    
    # ===== Analogical Types =====
    'Entity',
    'Relation',
    
    # ===== Causal Types =====
    'CausalEdge',
    
    # ===== Explainability =====
    'ReasoningExplainer',
    'SafetyAwareReasoning',
    
    # ===== Tool Selection (Optional) =====
    'ToolSelector',
    'SelectionRequest',
    'SelectionResult',
    'AdmissionControlIntegration',
    'RequestPriority',
    
    # ===== Bandit Learning (Optional) =====
    'AdaptiveBanditOrchestrator',
    'BanditContext',
    'BanditFeedback',
    'BanditAction',
    
    # ===== Availability Flags =====
    'PROBABILISTIC_AVAILABLE',
    'CAUSAL_AVAILABLE',
    'SYMBOLIC_AVAILABLE',
    'ANALOGICAL_AVAILABLE',
    'MULTIMODAL_AVAILABLE',
    'UNIFIED_AVAILABLE',
    'EXPLAINER_AVAILABLE',
    'SELECTION_AVAILABLE',
    'BANDIT_AVAILABLE',
]

# ============================================================================
# Module Status Report
# ============================================================================
def get_module_status() -> dict:
    """Get availability status of all reasoning components."""
    return {
        'probabilistic': PROBABILISTIC_AVAILABLE,
        'causal': CAUSAL_AVAILABLE,
        'symbolic': SYMBOLIC_AVAILABLE,
        'analogical': ANALOGICAL_AVAILABLE,
        'multimodal': MULTIMODAL_AVAILABLE,
        'unified': UNIFIED_AVAILABLE,
        'explainer': EXPLAINER_AVAILABLE,
        'selection': SELECTION_AVAILABLE,
        'bandit': BANDIT_AVAILABLE,
    }

def print_module_status():
    """Print a formatted status report of all components."""
    status = get_module_status()
    print("\n" + "="*60)
    print("VULCAN Reasoning Module Status")
    print("="*60)
    
    for component, available in status.items():
        status_icon = "✓" if available else "✗"
        status_text = "Available" if available else "Not Available"
        print(f"{status_icon} {component.capitalize():15s}: {status_text}")
    
    print("="*60)
    
    available_count = sum(status.values())
    total_count = len(status)
    print(f"Total: {available_count}/{total_count} components available")
    print("="*60 + "\n")

def create_unified_reasoner(config: Optional[dict] = None, 
                           enable_learning: bool = True,
                           enable_safety: bool = True) -> Optional['UnifiedReasoner']:
    """
    Convenience function to create a UnifiedReasoner with error handling.
    
    Args:
        config: Configuration dictionary
        enable_learning: Enable learning components
        enable_safety: Enable safety validation
        
    Returns:
        UnifiedReasoner instance or None if unavailable
    """
    if not UNIFIED_AVAILABLE:
        logger.error("Cannot create UnifiedReasoner - component not available")
        return None
    
    try:
        return UnifiedReasoner(
            config=config,
            enable_learning=enable_learning,
            enable_safety=enable_safety
        )
    except Exception as e:
        logger.error(f"Failed to create UnifiedReasoner: {e}")
        return None

# Export helpers
__all__.extend(['get_module_status', 'print_module_status', 'create_unified_reasoner'])

# ============================================================================
# Validation - Warn if critical components are missing
# ============================================================================
if not UNIFIED_AVAILABLE:
    logger.critical(
        "UnifiedReasoner not available! This is the main orchestrator. "
        "Most functionality will be unavailable."
    )

# Log successful initialization
available_reasoners = sum([
    PROBABILISTIC_AVAILABLE,
    CAUSAL_AVAILABLE,
    SYMBOLIC_AVAILABLE,
    ANALOGICAL_AVAILABLE,
    MULTIMODAL_AVAILABLE
])

logger.info(f"Vulcan Reasoning Module initialized: {available_reasoners}/5 reasoners available")

# ============================================================================
# Version Info
# ============================================================================
__version__ = '1.0.0'
__author__ = 'Vulcan AI Team'
__description__ = 'Unified reasoning system with multiple paradigms and adaptive tool selection'