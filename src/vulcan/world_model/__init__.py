"""Dependency-light world-model package facade.

Heavy research components are imported lazily so CSIU owners can be imported
without NumPy/Torch/FastAPI/aiohttp/NetworkX or the complete world-model graph.
"""
from __future__ import annotations
import importlib

__version__ = "0.1.0"
__author__ = "VULCAN-AGI Team"

_LAZY = {
    "WorldModel": ".world_model_core",
    "Observation": ".observation_types",
    "ModelContext": ".observation_types",
    "ComponentIntegrationError": ".observation_types",
    "NullMetaReasoningComponent": ".observation_types",
    "NullMotivationalIntrospection": ".observation_types",
    "NullObjectiveHierarchy": ".observation_types",
    "CausalDAG": ".causal_graph",
    "ConfidenceCalibrator": ".confidence_calibrator",
    "CorrelationTracker": ".correlation_tracker",
    "DynamicsModel": ".dynamics_model",
    "InterventionExecutor": ".intervention_manager",
    "InterventionPrioritizer": ".intervention_manager",
}
__all__ = tuple(_LAZY) + ("get_available_components", "check_dependencies", "get_module_info")

def __getattr__(name: str):
    mod = _LAZY.get(name)
    if not mod:
        raise AttributeError(name)
    value = getattr(importlib.import_module(mod, __name__), name)
    globals()[name] = value
    return value

def get_available_components():
    return {"world_model_core": True, "meta_reasoning": True}

def check_dependencies():
    import importlib.util
    return {k: importlib.util.find_spec(k) is not None for k in ("numpy","scipy","sklearn","pandas","networkx","statsmodels")}

def get_module_info():
    return {"version": __version__, "author": __author__, "components": get_available_components(), "dependencies": check_dependencies()}
