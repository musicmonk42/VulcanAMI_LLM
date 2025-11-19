"""
Semantic Bridge - Cross-domain knowledge transfer and concept mapping system
Part of the VULCAN-AGI system
"""

# Core components
from .semantic_bridge_core import (
    SemanticBridge,
    ConceptType,
    TransferStatus,
    PatternSignature,
    ConceptVersion,
    TransferCompatibility,
    ConceptConflict,
    retry_on_failure
)

from .concept_mapper import (
    ConceptMapper,
    Concept,
    PatternOutcome,
    MeasurableEffect,
    EffectType as MapperEffectType,
    GroundingStatus
)

from .conflict_resolver import (
    EvidenceWeightedResolver,
    ConflictResolution,
    ConflictType,
    ResolutionAction,
    Evidence,
    EvidenceType
)

from .domain_registry import (
    DomainRegistry,
    DomainProfile,
    DomainEffect,
    DomainCriticality,
    EffectCategory,
    Pattern,
    PatternType,
    RiskAdjuster,
    DomainRelationship
    # Removed: DomainCharacteristics - doesn't exist in domain_registry
)

from .transfer_engine import (
    TransferEngine,
    TransferDecision,
    TransferType,
    ConceptEffect,
    Mitigation,
    MitigationType,
    Constraint,
    ConstraintType,
    PartialTransferEngine,
    MitigationLearner,
    DomainCharacteristics,  # This one exists in transfer_engine
    EffectType as TransferEffectType
)

from .cache_manager import (
    CacheManager
)

# Define public API
__all__ = [
    # Core orchestrator
    'SemanticBridge',
    
    # Concept mapping
    'ConceptMapper',
    'Concept',
    'PatternOutcome',
    'MeasurableEffect',
    'MapperEffectType',
    'GroundingStatus',
    
    # Conflict resolution
    'EvidenceWeightedResolver',
    'ConflictResolution',
    'ConflictType',
    'ResolutionAction',
    'Evidence',
    'EvidenceType',
    
    # Domain management
    'DomainRegistry',
    'DomainProfile',
    'DomainEffect',
    'DomainCriticality',
    'EffectCategory',
    'Pattern',
    'PatternType',
    'RiskAdjuster',
    'DomainRelationship',
    
    # Transfer management
    'TransferEngine',
    'TransferDecision',
    'TransferType',
    'ConceptEffect',
    'Mitigation',
    'MitigationType',
    'Constraint',
    'ConstraintType',
    'PartialTransferEngine',
    'MitigationLearner',
    'DomainCharacteristics',  # From transfer_engine
    'TransferEffectType',
    
    # Cache management
    'CacheManager',
    
    # Utilities
    'retry_on_failure',
    'ConceptType',
    'TransferStatus',
    'PatternSignature',
    'ConceptVersion',
    'TransferCompatibility',
    'ConceptConflict',
]

# Module metadata
__version__ = '1.0.0'
__author__ = 'VULCAN-AGI Development Team'
__status__ = 'Production'

# Module configuration
DEFAULT_CONFIG = {
    'safety': {
        'max_risk_score': 0.8,
        'require_validation': True,
        'block_unsafe_transfers': True
    },
    'memory': {
        'cache_limit_mb': 1000,
        'max_concepts': 10000,
        'max_domains': 1000
    },
    'learning': {
        'min_evidence_count': 5,
        'confidence_threshold': 0.7,
        'enable_mitigation_learning': True
    },
    'transfer': {
        'full_transfer_threshold': 0.8,
        'partial_transfer_threshold': 0.5,
        'enable_rollback': True
    }
}


def get_default_config() -> dict:
    """
    Get default configuration for semantic bridge.
    
    Returns:
        Dictionary with default configuration values
    """
    return DEFAULT_CONFIG.copy()


def create_semantic_bridge(world_model=None, vulcan_memory=None, 
                           config: dict = None) -> SemanticBridge:
    """
    Factory function to create a configured SemanticBridge instance.
    
    Args:
        world_model: World model instance for causal reasoning
        vulcan_memory: VULCAN memory system for persistence
        config: Optional configuration dictionary (uses defaults if not provided)
    
    Returns:
        Configured SemanticBridge instance
    
    Example:
        >>> bridge = create_semantic_bridge(
        ...     world_model=my_world_model,
        ...     config={'safety': {'max_risk_score': 0.9}}
        ... )
    """
    if config is None:
        config = get_default_config()
    else:
        # Merge with defaults
        full_config = get_default_config()
        for key, value in config.items():
            if key in full_config and isinstance(value, dict):
                full_config[key].update(value)
            else:
                full_config[key] = value
        config = full_config
    
    return SemanticBridge(
        world_model=world_model,
        vulcan_memory=vulcan_memory,
        safety_config=config.get('safety')
    )


def get_version_info() -> dict:
    """
    Get detailed version and status information.
    
    Returns:
        Dictionary with version details
    """
    return {
        'version': __version__,
        'status': __status__,
        'components': {
            'semantic_bridge_core': 'Production',
            'concept_mapper': 'Production',
            'conflict_resolver': 'Production',
            'domain_registry': 'Production',
            'transfer_engine': 'Production',
            'cache_manager': 'Production'
        },
        'features': {
            'safety_validation': True,
            'world_model_integration': True,
            'bounded_data_structures': True,
            'mitigation_learning': True,
            'transfer_rollback': True,
            'adaptive_caching': True,
            'domain_adaptive_thresholds': True,
            'operation_history_persistence': True
        }
    }


# Module-level initialization logging
import logging
logger = logging.getLogger(__name__)
logger.info(
    "Semantic Bridge v%s initialized (%s) - All components production-ready",
    __version__, __status__
)