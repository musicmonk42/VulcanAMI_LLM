"""
Startup Phases

Defines the startup phase enumeration and phase-specific metadata.
Each phase represents a logical stage in the VULCAN-AGI initialization.
"""

from enum import Enum
from typing import Set, Optional
from dataclasses import dataclass


class StartupPhase(Enum):
    """
    Enumeration of startup phases in dependency order.
    
    Phases execute sequentially to ensure proper initialization order:
    1. CONFIGURATION - Load settings and config profiles
    2. CORE_SERVICES - Initialize deployment, LLM, Redis
    3. REASONING_SYSTEMS - Initialize reasoning subsystems
    4. MEMORY_SYSTEMS - Initialize memory subsystems  
    5. PRELOADING - Preload ML models and singletons
    6. MONITORING - Start memory guard, self-optimizer
    """
    
    CONFIGURATION = "configuration"
    CORE_SERVICES = "core_services"
    REASONING_SYSTEMS = "reasoning_systems"
    MEMORY_SYSTEMS = "memory_systems"
    PRELOADING = "preloading"
    MONITORING = "monitoring"


@dataclass
class PhaseMetadata:
    """
    Metadata for a startup phase.
    
    Attributes:
        name: Human-readable phase name
        critical: If True, phase failure prevents server startup
        timeout_seconds: Maximum time allowed for phase completion
        description: Brief description of phase purpose
    """
    
    name: str
    critical: bool
    timeout_seconds: float
    description: str


# Phase metadata configuration
PHASE_METADATA: dict[StartupPhase, PhaseMetadata] = {
    StartupPhase.CONFIGURATION: PhaseMetadata(
        name="Configuration Loading",
        critical=True,
        timeout_seconds=30,
        description="Load settings and configuration profiles"
    ),
    StartupPhase.CORE_SERVICES: PhaseMetadata(
        name="Core Services",
        critical=True,
        timeout_seconds=60,
        description="Initialize deployment, LLM, and Redis connection"
    ),
    StartupPhase.REASONING_SYSTEMS: PhaseMetadata(
        name="Reasoning Systems",
        critical=False,
        timeout_seconds=120,
        description="Initialize symbolic, probabilistic, causal, and abstract reasoning"
    ),
    StartupPhase.MEMORY_SYSTEMS: PhaseMetadata(
        name="Memory Systems",
        critical=False,
        timeout_seconds=60,
        description="Initialize long-term memory, episodic memory, and compressed memory"
    ),
    StartupPhase.PRELOADING: PhaseMetadata(
        name="Model Preloading",
        critical=False,
        timeout_seconds=180,
        description="Preload BERT, SentenceTransformer, and reasoning models"
    ),
    StartupPhase.MONITORING: PhaseMetadata(
        name="Monitoring Services",
        critical=False,
        timeout_seconds=30,
        description="Start memory guard and self-optimizer"
    ),
}


def get_phase_metadata(phase: StartupPhase) -> PhaseMetadata:
    """
    Get metadata for a specific phase.
    
    Args:
        phase: The startup phase
        
    Returns:
        PhaseMetadata for the phase
    """
    return PHASE_METADATA[phase]


def get_critical_phases() -> Set[StartupPhase]:
    """
    Get the set of critical phases that must succeed for startup.
    
    Returns:
        Set of critical StartupPhase values
    """
    return {phase for phase, meta in PHASE_METADATA.items() if meta.critical}


def is_critical_phase(phase: StartupPhase) -> bool:
    """
    Check if a phase is critical (failure prevents startup).
    
    Args:
        phase: The startup phase to check
        
    Returns:
        True if phase is critical, False otherwise
    """
    return PHASE_METADATA[phase].critical
