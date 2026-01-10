"""
Unified Reasoning Module (Re-export Shim)

This module provides backward compatibility by re-exporting everything from
the new modular unified/ package structure.

For new code, please import directly from vulcan.reasoning.unified:
    from vulcan.reasoning.unified import UnifiedReasoner
    
This file maintains backward compatibility for existing code:
    from vulcan.reasoning.unified_reasoning import UnifiedReasoner  # Still works!

The original 4,924-line monolithic file has been refactored into 9 focused modules:
- types.py (140 lines) - Core dataclasses
- config.py (159 lines) - Configuration constants
- component_loader.py (359 lines) - Lazy component loading
- cache.py (390 lines) - Weight management
- strategies.py (1,299 lines) - Strategy implementations  
- orchestrator.py (4,419 lines) - Main UnifiedReasoner class
- multimodal_handler.py (192 lines) - Multimodal methods
- persistence.py (142 lines) - State save/load
- __init__.py (109 lines) - Package exports

Total: 9 modules averaging 579 lines each (was 1 file with 4,924 lines)
"""

# Re-export everything from the unified package
from vulcan.reasoning.unified import *

# Explicitly re-export the main class for clarity
from vulcan.reasoning.unified import UnifiedReasoner

__all__ = [
    "UnifiedReasoner",
    "ReasoningTask",
    "ReasoningPlan",
    "ToolWeightManager",
    "compute_query_hash",
    # Strategy functions
    "_sequential_reasoning",
    "_parallel_reasoning",
    "_ensemble_reasoning",
    "_adaptive_reasoning",
    "_hybrid_reasoning",
    "_hierarchical_reasoning",
    "_portfolio_reasoning",
    "_utility_based_reasoning",
    # Multimodal functions
    "reason_multimodal",
    "reason_counterfactual",
    "reason_by_analogy",
    # Persistence functions
    "save_state",
    "load_state",
]
