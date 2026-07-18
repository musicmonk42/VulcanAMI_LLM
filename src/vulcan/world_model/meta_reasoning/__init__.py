"""Dependency-light meta-reasoning facade with lazy component imports."""
from __future__ import annotations
import importlib

_LAZY = {
    "SelfImprovementDrive": ".self_improvement_drive",
    "TriggerType": ".self_improvement_drive",
    "FailureType": ".self_improvement_drive",
    "ImprovementObjective": ".self_improvement_drive",
    "SelfImprovementState": ".self_improvement_drive",
    "compose_self_improvement_drive": ".self_improvement_drive",
    "CodeIntrospector": ".self_improvement_drive",
    "LogAnalyzer": ".self_improvement_drive",
    "CodeKnowledgeStore": ".self_improvement_drive",
    "CSIUEnforcement": ".csiu_enforcement",
    "CSIUEnforcementConfig": ".csiu_enforcement",
    "CSIUPolicy": ".csiu_enforcement",
    "CSIUMetricSnapshot": ".csiu_enforcement",
    "CSIUDecision": ".csiu_enforcement",
    "CSIUValidationError": ".csiu_enforcement",
    "canonical_digest": ".csiu_enforcement",
    "get_csiu_enforcer": ".csiu_enforcement",
    "GovernedSelfImprovementTransaction": ".governed_transaction",
    "ImprovementProposal": ".governed_transaction",
    "inspect_repository": ".governed_transaction",
}
__all__ = tuple(_LAZY)

def __getattr__(name: str):
    mod = _LAZY.get(name)
    if not mod:
        raise AttributeError(name)
    value = getattr(importlib.import_module(mod, __name__), name)
    globals()[name] = value
    return value
