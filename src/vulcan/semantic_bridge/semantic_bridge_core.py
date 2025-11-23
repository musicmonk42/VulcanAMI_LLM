"""
semantic_bridge_core.py - Main semantic bridge orchestrator
Part of the VULCAN-AGI system

Integrated with comprehensive safety validation.
FIXED: Component initialization with safety_config, world_model integration
ENHANCED: Pattern signature caching, operation history persistence, inverted index, retry logic, unified cache management
PRODUCTION-READY: All unbounded data structures fixed with proper limits and eviction
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque, Counter
import time
import json
import hashlib
from enum import Enum
from pathlib import Path
import copy
import threading
import functools

# Import safety validator
try:
    from ..safety.safety_validator import EnhancedSafetyValidator
    from ..safety.safety_types import SafetyConfig
    SAFETY_VALIDATOR_AVAILABLE = True
except ImportError:
    SAFETY_VALIDATOR_AVAILABLE = False
    # Note: Warning moved to __init__ to avoid spurious warnings at import time
    EnhancedSafetyValidator = None
    SafetyConfig = None

# Optional import with fallback
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logging.warning("networkx not available, using fallback graph implementation")
    
    # Simple fallback graph implementation
    class SimpleDiGraph:
        """Simple directed graph implementation for when NetworkX is not available"""
        def __init__(self):
            self.nodes_dict = {}
            self.edges_dict = defaultdict(list)
            self.edge_data = {}
            
        def add_node(self, node, **attrs):
            if node not in self.nodes_dict:
                self.nodes_dict[node] = attrs
            else:
                self.nodes_dict[node].update(attrs)
        
        def add_edge(self, source, target, **attrs):
            self.edges_dict[source].append(target)
            self.edge_data[(source, target)] = attrs
        
        def has_node(self, node):
            return node in self.nodes_dict
        
        def nodes(self):
            return list(self.nodes_dict.keys())
        
        def edges(self):
            edges = []
            for source, targets in self.edges_dict.items():
                for target in targets:
                    edges.append((source, target))
            return edges
        
        def predecessors(self, node):
            preds = []
            for source, targets in self.edges_dict.items():
                if node in targets:
                    preds.append(source)
            return preds
        
        def successors(self, node):
            return self.edges_dict.get(node, [])
    
    nx = type('nx', (), {'DiGraph': SimpleDiGraph})()

# Import from other semantic_bridge modules
try:
    from .concept_mapper import ConceptMapper, Concept, PatternOutcome
    from .conflict_resolver import EvidenceWeightedResolver
    from .domain_registry import DomainRegistry
    from .transfer_engine import TransferEngine
    from .cache_manager import CacheManager
except ImportError as e:
    logging.warning(f"Failed to import semantic_bridge components: {e}")
    # Provide basic fallback classes
    
    class Concept:
        def __init__(self, pattern_signature, grounded_effects, confidence):
            self.concept_id = f"concept_{hashlib.md5(pattern_signature.encode()).hexdigest()[:8]}"
            self.pattern_signature = pattern_signature
            self.grounded_effects = grounded_effects
            self.confidence = confidence
            self.domains = set()
            self.usage_count = 0
            self.success_rate = 0.5
            self.metadata = {}
            
        def update_usage(self, success: bool):
            self.usage_count += 1
            alpha = 0.1
            self.success_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * self.success_rate
            
        def get_signature(self):
            return self.pattern_signature
    
    class PatternOutcome:
        def __init__(self, pattern_id, success, domain, execution_time, **kwargs):
            self.pattern_id = pattern_id
            self.success = success
            self.domain = domain
            self.execution_time = execution_time
            self.metrics = kwargs.get('metrics', {})
            self.errors = kwargs.get('errors', [])
            self.context = kwargs.get('context', {})
    
    class ConceptMapper:
        def __init__(self, safety_config=None, world_model=None):
            self.concepts = {}
            self.safety_config = safety_config
            
        def map_pattern_to_concept(self, pattern, domain="general"):
            pattern_sig = hashlib.md5(str(pattern).encode()).hexdigest()
            if pattern_sig not in self.concepts:
                concept = Concept(pattern_sig, [], 0.5)
                concept.domains.add(domain)
                self.concepts[pattern_sig] = concept
            return self.concepts[pattern_sig]
            
        def register_concept(self, concept):
            self.concepts[concept.concept_id] = concept
            
        def find_similar_concepts(self, concept, top_k=5):
            similar = []
            for cid, other in self.concepts.items():
                if cid != concept.concept_id:
                    if hasattr(concept, 'domains') and hasattr(other, 'domains'):
                        overlap = len(concept.domains & other.domains) / max(1, len(concept.domains | other.domains))
                        similar.append((other, overlap))
            similar.sort(key=lambda x: x[1], reverse=True)
            return similar[:top_k]
    
    class EvidenceWeightedResolver:
        def __init__(self, safety_config=None, world_model=None):
            self.resolution_history = deque(maxlen=100)
            self.safety_config = safety_config
            
        def resolve_conflict(self, conflict):
            return {'action': 'coexist', 'winner': None, 'confidence': 0.5, 'reasoning': []}
    
    class DomainRegistry:
        def __init__(self, safety_config=None, world_model=None):
            self.domains = {}
            self.safety_config = safety_config
            
        def register_domain(self, name, characteristics=None):
            self.domains[name] = {'name': name, 'characteristics': characteristics or {}}
            
        def get_related_domains(self, domain):
            return []
            
        def update_domain_performance(self, domain, success):
            pass
    
    class TransferEngine:
        def __init__(self, safety_config=None, world_model=None):
            self.transfer_history = deque(maxlen=100)
            self.safety_config = safety_config
            
        def validate_full_transfer(self, concept, source, target):
            @dataclass
            class TransferDecision:
                type: str = "full"
                confidence: float = 0.5
                mitigations: list = field(default_factory=list)
                constraints: list = field(default_factory=list)
                reasoning: list = field(default_factory=list)
                risk_assessment: dict = field(default_factory=dict)
                estimated_cost: float = 0.0
                
                def is_transferable(self):
                    return self.confidence >= 0.5
            
            return TransferDecision()
        
        def execute_transfer(self, concept, transfer_decision, target_domain):
            new_concept = copy.deepcopy(concept)
            if hasattr(new_concept, 'domains'):
                if isinstance(new_concept.domains, set):
                    new_concept.domains.add(target_domain)
                else:
                    new_concept.domains = {target_domain}
            return {'success': True, 'transferred_concept': new_concept}
    
    class CacheManager:
        """Fallback cache manager if import fails"""
        def __init__(self, max_memory_mb=1000):
            self.max_memory = max_memory_mb
            
        def register_cache(self, name, cache, priority=5, clear_callback=None):
            pass
            
        def check_memory(self):
            return {'total_mb': 0, 'limit_mb': self.max_memory}
            
        def record_hit(self, cache_name):
            pass
            
        def record_miss(self, cache_name):
            pass
            
        def get_statistics(self):
            return {}

logger = logging.getLogger(__name__)


def retry_on_failure(max_attempts=3, delay=1.0):
    """
    Decorator for retrying operations (FIXED: retry logic)
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Base delay between retries (increases with each attempt)
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.debug("Attempt %d failed for %s, retrying: %s", 
                                   attempt + 1, func.__name__, e)
                        time.sleep(delay * (attempt + 1))
                    else:
                        logger.error("All %d attempts failed for %s: %s", 
                                   max_attempts, func.__name__, e)
            raise last_exception
        return wrapper
    return decorator


class ConceptType(Enum):
    """Types of concepts"""
    STRUCTURAL = "structural"
    FUNCTIONAL = "functional"
    BEHAVIORAL = "behavioral"
    COMPOSITIONAL = "compositional"
    ABSTRACT = "abstract"


class TransferStatus(Enum):
    """Status of concept transfer"""
    COMPATIBLE = "compatible"
    PARTIAL = "partial"
    INCOMPATIBLE = "incompatible"
    REQUIRES_ADAPTATION = "requires_adaptation"


@dataclass
class PatternSignature:
    """Signature characterizing a pattern for operation selection"""
    is_novel: bool
    success_rate: float
    confidence: float
    evidence_count: int
    domains_count: int
    has_conflicts: bool
    is_significant_change: bool
    pattern_complexity: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'is_novel': self.is_novel,
            'success_rate': self.success_rate,
            'confidence': self.confidence,
            'evidence_count': self.evidence_count,
            'domains_count': self.domains_count,
            'has_conflicts': self.has_conflicts,
            'is_significant_change': self.is_significant_change,
            'pattern_complexity': self.pattern_complexity
        }


@dataclass
class ConceptVersion:
    """Version of a concept"""
    version_id: str
    concept_id: str
    version_number: int
    changes: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    performance: Dict[str, float] = field(default_factory=dict)
    parent_version: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'version_id': self.version_id,
            'concept_id': self.concept_id,
            'version_number': self.version_number,
            'changes': self.changes,
            'timestamp': self.timestamp,
            'performance': self.performance,
            'parent_version': self.parent_version
        }


@dataclass
class TransferCompatibility:
    """Compatibility assessment for concept transfer"""
    source_domain: str
    target_domain: str
    concept_id: str
    compatibility_score: float
    required_adaptations: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    confidence: float = 0.5
    
    def is_compatible(self) -> bool:
        """Check if transfer is compatible"""
        return self.compatibility_score >= 0.6 and len(self.risks) == 0


@dataclass
class ConceptConflict:
    """Represents a conflict between concepts"""
    new_concept: Any
    existing_concept: Any
    conflict_type: str
    severity: float
    resolution_options: List[str] = field(default_factory=list)


class SemanticBridge:
    """Main semantic bridge orchestrator - Integrated with Safety Validation and World Model"""
    
    def __init__(self, 
                 world_model=None,
                 vulcan_memory=None,
                 safety_config: Optional[Dict[str, Any]] = None):
        """
        Initialize semantic bridge - FIXED: Added world_model integration
        
        Args:
            world_model: World model instance for accessing causal knowledge
            vulcan_memory: VULCAN memory system for storage
            safety_config: Optional safety configuration
        """
        self.world_model = world_model
        self.memory = vulcan_memory
        
        # Initialize safety validator
        if SAFETY_VALIDATOR_AVAILABLE:
            if isinstance(safety_config, dict) and safety_config:
                self.safety_validator = EnhancedSafetyValidator(SafetyConfig.from_dict(safety_config))
            else:
                self.safety_validator = EnhancedSafetyValidator()
            logger.info("SemanticBridge: Safety validator initialized")
        else:
            self.safety_validator = None
            logger.warning("SemanticBridge: Safety validator not available - operating without safety checks")
        
        # FIXED: Pass safety_config and world_model to all components
        self.concept_mapper = ConceptMapper(world_model=world_model, safety_config=safety_config)
        self.conflict_resolver = EvidenceWeightedResolver(world_model=world_model, safety_config=safety_config)
        self.transfer_engine = TransferEngine(world_model=world_model, safety_config=safety_config)
        self.domain_registry = DomainRegistry(world_model=world_model, safety_config=safety_config)
        
        # FIXED: Concept versioning with size limits
        self.concept_versions = {}  # concept_id -> list of ConceptVersion
        self.max_versioned_concepts = 5000
        self.max_versions_per_concept = 20
        
        self.current_versions = {}  # concept_id -> current version_id
        self.max_current_versions = 10000
        
        # Graph for concept relationships
        if NETWORKX_AVAILABLE:
            self.concept_graph = nx.DiGraph()
        else:
            self.concept_graph = SimpleDiGraph()
        
        # Statistics
        self.total_concepts = 0
        self.total_transfers = 0
        self.total_conflicts = 0
        
        # FIXED: Replace defaultdict(int) with Counter
        self.safety_blocks = Counter()
        self.safety_corrections = Counter()
        
        # FIXED: Caches with explicit size limits
        self.pattern_signature_cache = {}
        self.max_pattern_cache_size = 2000
        
        self.domain_concept_cache = {}
        self.max_domain_cache_size = 1000
        
        self.cache_ttl = 300  # 5 minutes
        self.last_cache_clear = time.time()
        
        # FIXED: Inverted index with size limits
        self.domain_concept_index = {}  # Changed from defaultdict to regular dict
        self.max_index_domains = 1000
        self.max_concepts_per_domain = 10000
        self.index_dirty = False
        
        # Operation selection tracking
        self.operation_history = deque(maxlen=500)
        
        # FIXED: Persistence tracking
        self._last_history_persist = time.time()
        self._history_persist_interval = 300  # 5 minutes
        
        # Thread safety
        self._concept_lock = threading.RLock()
        
        # FIXED: Create unified cache manager
        self.cache_manager = CacheManager(max_memory_mb=1000)
        
        # Register all caches
        self.cache_manager.register_cache('pattern_signature', self.pattern_signature_cache, priority=7)
        self.cache_manager.register_cache('domain_concept', self.domain_concept_cache, priority=9)
        
        # Register component caches
        if hasattr(self.domain_registry, 'distance_cache'):
            self.cache_manager.register_cache('domain_distance', 
                                             self.domain_registry.distance_cache, 
                                             priority=6)
        
        if hasattr(self.transfer_engine, 'compatibility_cache'):
            self.cache_manager.register_cache('transfer_compatibility',
                                             self.transfer_engine.compatibility_cache,
                                             priority=8)
        
        # Initialize default domains
        self._initialize_default_domains()
        
        # FIXED: Load operation history from disk
        self._load_operation_history()
        
        logger.info("SemanticBridge initialized (production-ready) with bounded data structures")
    
    def _evict_oldest_versioned_concept(self):
        """
        Evict concept with oldest versions (FIXED: version size limit)
        """
        if not self.concept_versions:
            return
        
        # Find concept with oldest average version timestamp
        oldest_concept = None
        oldest_avg_time = float('inf')
        
        for concept_id, versions in self.concept_versions.items():
            if versions:
                avg_time = sum(v.timestamp for v in versions) / len(versions)
                if avg_time < oldest_avg_time:
                    oldest_avg_time = avg_time
                    oldest_concept = concept_id
        
        if oldest_concept:
            del self.concept_versions[oldest_concept]
            if oldest_concept in self.current_versions:
                del self.current_versions[oldest_concept]
            logger.debug("Evicted versions for concept %s", oldest_concept)
    
    def _evict_oldest_index_domain(self):
        """
        Evict domain with fewest concepts from index (FIXED: index size limit)
        """
        if not self.domain_concept_index:
            return
        
        # Find domain with fewest concepts
        min_domain = min(self.domain_concept_index.keys(),
                        key=lambda k: len(self.domain_concept_index[k]))
        
        del self.domain_concept_index[min_domain]
        logger.debug("Evicted domain %s from concept index", min_domain)
    
    def learn_concept_from_pattern(self, pattern, outcomes: List[PatternOutcome]) -> Optional[Concept]:
        """
        Learn concept from pattern and its outcomes - WITH SAFETY VALIDATION
        
        CRITICAL: This method bridges execution to knowledge. Safety validation is strongly
        recommended to prevent learning from unsafe patterns or corrupted outcomes.
        
        Args:
            pattern: Pattern to learn from
            outcomes: Execution outcomes
            
        Returns:
            Learned concept or None if unsafe
        """
        start_time = time.time()
        
        # SAFETY WARNING: If no safety validator, log warning but continue
        if self.safety_validator is None:
            logger.warning(
                "SAFETY WARNING: learn_concept_from_pattern called without safety_validator. "
                "Operating in degraded safety mode - no pattern/outcome validation will occur."
            )
            self.safety_blocks['no_validator_available'] += 1
            safe_outcomes = outcomes
        else:
            # SAFETY: Validate pattern before learning
            pattern_check = self.safety_validator.validate_pattern(pattern)
            if not pattern_check['safe']:
                logger.warning("Blocked unsafe pattern: %s", pattern_check['reason'])
                self.safety_blocks['pattern'] += 1
                return None
            
            # SAFETY: Filter to safe outcomes only
            safe_outcomes = []
            for outcome in outcomes:
                outcome_check = self.safety_validator.analyze_outcome_safety(outcome)
                if outcome_check['safe']:
                    safe_outcomes.append(outcome)
                else:
                    logger.debug("Filtered unsafe outcome: %s", outcome_check['reason'])
                    self.safety_blocks['outcome'] += 1
        
        if not safe_outcomes:
            logger.warning("No safe outcomes available for learning")
            self.safety_blocks['no_safe_outcomes'] += 1
            return None
        
        # EXAMINE: Extract pattern signature from SAFE outcomes only
        signature = self._extract_pattern_signature(pattern, safe_outcomes)
        
        # SAFETY: Validate signature
        if self.safety_validator:
            sig_validation = self._validate_signature_safety(signature)
            if not sig_validation['safe']:
                logger.warning("Unsafe signature detected: %s", sig_validation['reason'])
                self.safety_corrections['signature'] += 1
                # Apply corrections
                signature = self._apply_signature_corrections(signature)
        
        # SELECT: Decide which operations to run
        operations = self._select_operations(signature, pattern, safe_outcomes)
        
        logger.debug("Selected operations for pattern: %s", operations)
        
        # APPLY: Execute selected operations
        concept = None
        
        if 'map' in operations:
            # Map pattern to concept
            domains = set(o.domain for o in safe_outcomes)
            with self._concept_lock:
                concept = self.concept_mapper.map_pattern_to_concept(
                    pattern, 
                    list(domains)[0] if domains else "general"
                )
                # FIXED: Mark index as dirty when concepts change
                self.index_dirty = True
        else:
            # Skip mapping - pattern not worth learning
            logger.debug("Skipping concept mapping for low-value pattern")
            return None
        
        if 'update_metrics' in operations:
            # Update concept based on outcomes
            success_rate = signature.success_rate
            domains = set(o.domain for o in safe_outcomes)
            
            with self._concept_lock:
                if hasattr(concept, 'success_rate'):
                    concept.success_rate = success_rate
                if hasattr(concept, 'domains'):
                    if isinstance(concept.domains, set):
                        concept.domains.update(domains)
                    else:
                        concept.domains = domains
                
                # Extract features from outcomes
                if not hasattr(concept, 'features'):
                    concept.features = {}
                        
                for outcome in safe_outcomes:
                    if outcome.success and hasattr(outcome, 'metrics') and outcome.metrics:
                        for key, value in outcome.metrics.items():
                            if key not in concept.features:
                                concept.features[key] = []
                            concept.features[key].append(value)
                
                # Average features
                for key in concept.features:
                    if isinstance(concept.features[key], list) and concept.features[key]:
                        concept.features[key] = np.mean(concept.features[key])
        
        if 'version' in operations:
            # Create version
            self._create_concept_version(concept, changes=["Initial creation from pattern"])
        
        if 'graph' in operations:
            # Add to concept graph
            self.concept_graph.add_node(
                concept.concept_id if hasattr(concept, 'concept_id') else str(concept), 
                concept=concept
            )
        
        if 'storage' in operations:
            # Store in memory if available
            if self.memory:
                try:
                    self.memory.store_concept(concept)
                except Exception as e:
                    logger.debug("Failed to store concept in memory: %s", e)
        
        if 'world_model_link' in operations:
            # FIXED: Link concept to world model causal graph if available
            if self.world_model and hasattr(self.world_model, 'causal_graph'):
                try:
                    self._link_concept_to_world_model(concept, safe_outcomes)
                except Exception as e:
                    logger.debug("Failed to link concept to world model: %s", e)
        
        # REMEMBER: Track operation execution
        self.operation_history.append({
            'pattern_signature': signature,
            'operations_run': operations,
            'success': True,
            'execution_time': time.time() - start_time,
            'timestamp': time.time(),
            'safe_outcomes_count': len(safe_outcomes),
            'filtered_outcomes_count': len(outcomes) - len(safe_outcomes)
        })
        
        # FIXED: Periodic persistence
        if time.time() - self._last_history_persist > self._history_persist_interval:
            self._persist_operation_history()
            self._last_history_persist = time.time()
        
        self.total_concepts += 1
        
        logger.info("Learned concept from pattern (ops: %s, success_rate: %.2f, safety_filtered: %d)", 
                   operations, signature.success_rate, len(outcomes) - len(safe_outcomes))
        
        return concept
    
    @retry_on_failure(max_attempts=3, delay=0.5)
    def _link_concept_to_world_model(self, concept: Concept, outcomes: List[PatternOutcome]):
        """
        Link concept to world model causal graph with retry logic (FIXED: retry decorator)
        
        Args:
            concept: Concept to link
            outcomes: Outcomes that provide causal evidence
        """
        if not self.world_model:
            return
        
        # Extract causal relationships from outcomes
        for outcome in outcomes:
            if not hasattr(outcome, 'context') or not outcome.context:
                continue
            
            context = outcome.context
            
            # Look for action -> effect relationships
            if 'action' in context and 'effects' in context:
                action = context['action']
                effects = context['effects']
                
                # Add to world model's causal graph
                for effect in effects:
                    if isinstance(effect, dict) and 'variable' in effect:
                        effect_var = effect['variable']
                        strength = effect.get('strength', 0.5)
                        
                        # Validate with safety
                        if self.safety_validator:
                            try:
                                if hasattr(self.safety_validator, 'validate_causal_edge'):
                                    edge_validation = self.safety_validator.validate_causal_edge(
                                        action, effect_var, strength
                                    )
                                    if not edge_validation.get('safe', True):
                                        continue
                            except Exception as e:
                                logger.error("Safety validation error: %s", e)
                                continue
                        
                        # Add edge to causal graph
                        try:
                            self.world_model.causal_graph.add_edge(
                                action,
                                effect_var,
                                strength=strength,
                                evidence_type="semantic_bridge"
                            )
                            logger.debug("Linked concept to world model: %s -> %s", action, effect_var)
                        except Exception as e:
                            logger.debug("Failed to add edge to world model: %s", e)
    
    def _persist_operation_history(self):
        """
        Persist operation history to disk (FIXED: persistence)
        """
        if not self.memory or not hasattr(self.memory, 'storage_path'):
            return
        
        try:
            storage_path = Path(self.memory.storage_path) / 'semantic_bridge'
            storage_path.mkdir(parents=True, exist_ok=True)
            
            history_file = storage_path / 'operation_history.jsonl'
            
            # Append new entries
            with open(history_file, 'a') as f:
                for entry in list(self.operation_history):
                    # Convert to JSON-serializable format
                    serialized = {
                        'pattern_signature': entry['pattern_signature'].to_dict() if hasattr(entry['pattern_signature'], 'to_dict') else str(entry['pattern_signature']),
                        'operations_run': list(entry['operations_run']),
                        'success': entry['success'],
                        'execution_time': entry['execution_time'],
                        'timestamp': entry['timestamp'],
                        'safe_outcomes_count': entry.get('safe_outcomes_count', 0),
                        'filtered_outcomes_count': entry.get('filtered_outcomes_count', 0)
                    }
                    f.write(json.dumps(serialized) + '\n')
            
            logger.debug("Persisted %d operation history entries", len(self.operation_history))
            
        except Exception as e:
            logger.error("Failed to persist operation history: %s", e)
    
    def _load_operation_history(self):
        """
        Load operation history from disk (FIXED: persistence)
        """
        if not self.memory or not hasattr(self.memory, 'storage_path'):
            return
        
        try:
            storage_path = Path(self.memory.storage_path) / 'semantic_bridge'
            history_file = storage_path / 'operation_history.jsonl'
            
            if not history_file.exists():
                return
            
            with open(history_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        self.operation_history.append(entry)
                    except json.JSONDecodeError:
                        continue
            
            logger.info("Loaded %d operation history entries", len(self.operation_history))
            
        except Exception as e:
            logger.error("Failed to load operation history: %s", e)
    
    def _rebuild_domain_concept_index(self):
        """
        Rebuild inverted index for domain-concept mapping (FIXED: inverted index)
        """
        with self._concept_lock:
            self.domain_concept_index.clear()
            
            for concept_id, concept in self.concept_mapper.concepts.items():
                if hasattr(concept, 'domains'):
                    domains = concept.domains if isinstance(concept.domains, (set, list)) else [concept.domains]
                    for domain in domains:
                        # FIXED: Enforce index limits
                        if domain not in self.domain_concept_index:
                            if len(self.domain_concept_index) >= self.max_index_domains:
                                self._evict_oldest_index_domain()
                            self.domain_concept_index[domain] = set()
                        
                        # FIXED: Limit concepts per domain
                        if len(self.domain_concept_index[domain]) < self.max_concepts_per_domain:
                            self.domain_concept_index[domain].add(concept_id)
            
            self.index_dirty = False
            logger.debug("Rebuilt domain-concept index with %d domains", 
                        len(self.domain_concept_index))
    
    def get_world_model_insights(self, concept: Concept) -> Dict[str, Any]:
        """
        Get insights from world model for a concept - FIXED: New method for integration
        
        Args:
            concept: Concept to get insights for
            
        Returns:
            Dictionary with world model insights
        """
        if not self.world_model:
            return {'available': False}
        
        insights = {'available': True}
        
        # Get causal paths from world model
        if hasattr(self.world_model, 'causal_graph'):
            try:
                # Extract variables from concept
                concept_vars = []
                if hasattr(concept, 'grounded_effects'):
                    for effect in concept.grounded_effects:
                        if isinstance(effect, dict) and 'variable' in effect:
                            concept_vars.append(effect['variable'])
                
                # Find causal paths
                if concept_vars and hasattr(self.world_model.causal_graph, 'find_all_paths'):
                    paths = []
                    for var in concept_vars[:5]:  # Limit to first 5
                        try:
                            var_paths = self.world_model.causal_graph.find_all_paths(var, concept_vars)
                            paths.extend(var_paths)
                        except:
                            pass
                    
                    insights['causal_paths'] = len(paths)
                    insights['causal_depth'] = max([len(p.nodes) if hasattr(p, 'nodes') else 0 for p in paths], default=0)
                
            except Exception as e:
                logger.debug("Error getting world model insights: %s", e)
                insights['error'] = str(e)
        
        # Get correlation data
        if hasattr(self.world_model, 'correlation_tracker'):
            try:
                if hasattr(concept, 'domains') and concept.domains:
                    domain = list(concept.domains)[0] if isinstance(concept.domains, set) else concept.domains
                    insights['domain'] = domain
            except Exception as e:
                logger.debug("Error getting correlation data: %s", e)
        
        return insights
    
    def _extract_pattern_signature(self, pattern, outcomes: List[PatternOutcome]) -> PatternSignature:
        """
        Extract signature with cache invalidation on outcome changes (FIXED: cache staleness)
        
        Args:
            pattern: Pattern to analyze
            outcomes: Execution outcomes
            
        Returns:
            Pattern signature for operation selection
        """
        pattern_str = str(pattern)
        pattern_hash = hashlib.md5(pattern_str.encode()).hexdigest()
        
        # FIXED: Create cache key that includes outcome count to prevent staleness
        cache_key = f"{pattern_hash}_{len(outcomes)}"
        
        if cache_key in self.pattern_signature_cache:
            cached_sig, cached_time = self.pattern_signature_cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                self.cache_manager.record_hit('pattern_signature')
                return cached_sig
            else:
                # Expired - remove
                del self.pattern_signature_cache[cache_key]
        
        self.cache_manager.record_miss('pattern_signature')
        
        # Calculate metrics
        success_rate = sum(1 for o in outcomes if o.success) / max(1, len(outcomes))
        domains = set(o.domain for o in outcomes)
        
        # Check if pattern exists
        pattern_sig_str = getattr(pattern, 'pattern_signature', pattern_str)
        is_novel = pattern_sig_str not in self.concept_mapper.concepts
        
        # Check for conflicts
        has_conflicts = False
        if not is_novel:
            # Pattern exists - might have conflicts
            existing_concept = self.concept_mapper.concepts.get(pattern_sig_str)
            if existing_concept:
                if hasattr(existing_concept, 'success_rate'):
                    # Significant difference in performance = conflict
                    if abs(existing_concept.success_rate - success_rate) > 0.3:
                        has_conflicts = True
        
        # Determine if change is significant
        is_significant_change = (
            success_rate > 0.7 or success_rate < 0.3 or  # Very good or very bad
            len(outcomes) >= 10 or  # Enough evidence
            len(domains) > 2  # Multi-domain
        )
        
        # Calculate pattern complexity
        pattern_complexity = self._estimate_pattern_complexity(pattern, outcomes)
        
        # Create signature
        signature = PatternSignature(
            is_novel=is_novel,
            success_rate=success_rate,
            confidence=success_rate if success_rate > 0.5 else 0.5,
            evidence_count=len(outcomes),
            domains_count=len(domains),
            has_conflicts=has_conflicts,
            is_significant_change=is_significant_change,
            pattern_complexity=pattern_complexity,
            metadata={
                'domains': list(domains),
                'pattern_type': type(pattern).__name__
            }
        )
        
        # FIXED: Cache with size limit
        if len(self.pattern_signature_cache) < self.max_pattern_cache_size:
            self.pattern_signature_cache[cache_key] = (signature, time.time())
        else:
            # Evict oldest entry
            oldest_key = min(self.pattern_signature_cache.keys(),
                           key=lambda k: self.pattern_signature_cache[k][1])
            del self.pattern_signature_cache[oldest_key]
            self.pattern_signature_cache[cache_key] = (signature, time.time())
        
        return signature
    
    def _select_operations(self, signature: PatternSignature, 
                          pattern: Any, outcomes: List[PatternOutcome]) -> Set[str]:
        """
        Select which operations to run
        
        Args:
            signature: Pattern signature
            pattern: Pattern to process
            outcomes: Execution outcomes
            
        Returns:
            Set of operation names to execute
        """
        operations = set()
        
        # RULE 1: Always map if novel and has evidence
        if signature.is_novel and signature.evidence_count > 0:
            operations.add('map')
            operations.add('update_metrics')
            logger.debug("Novel pattern: mapping and updating metrics")
        
        # RULE 2: Skip mapping if not novel and no significant change
        elif not signature.is_novel and not signature.is_significant_change:
            logger.debug("Existing pattern with no significant change: skipping all operations")
            return operations  # Return empty set
        
        # RULE 3: Map if significant change (even if not novel)
        elif signature.is_significant_change:
            operations.add('map')
            operations.add('update_metrics')
            logger.debug("Significant change detected: mapping and updating metrics")
        
        # RULE 4: Only version if confident and significant
        if signature.confidence > 0.7 and signature.is_significant_change:
            operations.add('version')
            logger.debug("High confidence + significant change: creating version")
        
        # RULE 5: Only add to graph if confident
        if signature.confidence > 0.7 and signature.evidence_count >= 5:
            operations.add('graph')
            logger.debug("High confidence + sufficient evidence: adding to graph")
        
        # RULE 6: Store in memory if novel and good performance
        if signature.is_novel and signature.success_rate > 0.6:
            operations.add('storage')
            logger.debug("Novel with good performance: storing in memory")
        
        # RULE 7: Update existing if has conflicts
        if signature.has_conflicts:
            operations.add('update_metrics')
            operations.add('version')
            logger.debug("Conflicts detected: updating metrics and versioning")
        
        # RULE 8: Skip if very low evidence and low confidence
        if signature.evidence_count < 3 and signature.confidence < 0.5:
            logger.debug("Insufficient evidence and confidence: skipping all operations")
            return set()  # Skip everything
        
        # RULE 9: For complex patterns, be more selective
        if signature.pattern_complexity > 0.7:
            # Only do essential operations for complex patterns
            essential_ops = {'map', 'update_metrics'}
            operations = operations & essential_ops
            logger.debug("Complex pattern: limiting to essential operations")
        
        # RULE 10: FIXED - Link to world model if available and significant
        if self.world_model and signature.is_significant_change and signature.confidence > 0.6:
            operations.add('world_model_link')
            logger.debug("Significant pattern with world model: linking to causal graph")
        
        return operations
    
    def _estimate_pattern_complexity(self, pattern: Any, outcomes: List[PatternOutcome]) -> float:
        """
        Estimate pattern complexity
        
        Args:
            pattern: Pattern to analyze
            outcomes: Execution outcomes
            
        Returns:
            Complexity score (0-1)
        """
        complexity = 0.0
        
        # Factor 1: Number of domains
        domains = set(o.domain for o in outcomes)
        complexity += min(0.3, len(domains) * 0.1)
        
        # Factor 2: Variance in performance
        if outcomes:
            success_values = [1.0 if o.success else 0.0 for o in outcomes]
            if len(success_values) > 1:
                variance = np.var(success_values)
                complexity += variance * 0.3
        
        # Factor 3: Feature count
        if hasattr(pattern, 'features'):
            feature_count = len(pattern.features) if hasattr(pattern.features, '__len__') else 0
            complexity += min(0.2, feature_count * 0.02)
        
        # Factor 4: Execution time variance
        if outcomes:
            times = [o.execution_time for o in outcomes if hasattr(o, 'execution_time')]
            if len(times) > 1:
                time_variance = np.var(times) / (np.mean(times) + 1e-10)
                complexity += min(0.2, time_variance * 0.1)
        
        return min(1.0, complexity)
    
    def _validate_signature_safety(self, signature: PatternSignature) -> Dict[str, Any]:
        """Validate signature for safety issues"""
        violations = []
        
        # Check confidence bounds
        if signature.confidence < 0 or signature.confidence > 1:
            violations.append(f"Invalid confidence: {signature.confidence}")
        
        # Check success rate bounds
        if signature.success_rate < 0 or signature.success_rate > 1:
            violations.append(f"Invalid success rate: {signature.success_rate}")
        
        # Check evidence count
        if signature.evidence_count < 0:
            violations.append(f"Invalid evidence count: {signature.evidence_count}")
        
        # Check complexity bounds
        if signature.pattern_complexity < 0 or signature.pattern_complexity > 1:
            violations.append(f"Invalid pattern complexity: {signature.pattern_complexity}")
        
        if violations:
            return {'safe': False, 'reason': '; '.join(violations)}
        
        return {'safe': True}
    
    def _apply_signature_corrections(self, signature: PatternSignature) -> PatternSignature:
        """Apply safety corrections to signature"""
        # Clamp values to valid ranges
        signature.confidence = np.clip(signature.confidence, 0, 1)
        signature.success_rate = np.clip(signature.success_rate, 0, 1)
        signature.evidence_count = max(0, signature.evidence_count)
        signature.pattern_complexity = np.clip(signature.pattern_complexity, 0, 1)
        signature.domains_count = max(0, signature.domains_count)
        
        return signature
    
    def select_transfer_strategy(self, pattern: Any, target_domain: str) -> str:
        """
        Select optimal transfer strategy - ROUTER LOGIC
        
        Args:
            pattern: Pattern to transfer
            target_domain: Target domain
            
        Returns:
            Transfer strategy name
        """
        # EXAMINE pattern characteristics
        is_structural = self._is_structural_pattern(pattern)
        is_functional = self._is_functional_pattern(pattern)
        
        # Get source domain
        source_domain = None
        if hasattr(pattern, 'source_domain'):
            source_domain = pattern.source_domain
        elif hasattr(pattern, 'domains') and pattern.domains:
            source_domain = list(pattern.domains)[0] if isinstance(pattern.domains, (set, list)) else pattern.domains
        
        if source_domain:
            domain_similarity = self._calculate_domain_similarity(source_domain, target_domain)
        else:
            domain_similarity = 0.5  # Unknown similarity
        
        # SELECT strategy based on examination
        if domain_similarity > 0.8:
            strategy = 'direct_transfer'  # Domains very similar
            logger.debug("Selected direct_transfer strategy (similarity: %.2f)", domain_similarity)
        elif is_structural:
            strategy = 'structural_analogy'
            logger.debug("Selected structural_analogy strategy")
        elif is_functional:
            strategy = 'functional_transfer'
            logger.debug("Selected functional_transfer strategy")
        else:
            strategy = 'general_analogy'
            logger.debug("Selected general_analogy strategy")
        
        return strategy
    
    def _is_structural_pattern(self, pattern: Any) -> bool:
        """Check if pattern is primarily structural"""
        if hasattr(pattern, 'type') and pattern.type == ConceptType.STRUCTURAL:
            return True
        if hasattr(pattern, 'structure') and pattern.structure is not None:
            return True
        return False
    
    def _is_functional_pattern(self, pattern: Any) -> bool:
        """Check if pattern is primarily functional"""
        if hasattr(pattern, 'type') and pattern.type == ConceptType.FUNCTIONAL:
            return True
        if hasattr(pattern, 'function') and pattern.function is not None:
            return True
        return False
    
    def _get_domain_characteristics(self, domain_name: str) -> Dict[str, Any]:
        """
        Get domain characteristics, handling both dict and DomainProfile storage (FIXED: DomainProfile compatibility)
        
        Args:
            domain_name: Name of the domain
            
        Returns:
            Dictionary of characteristics
        """
        domain_obj = self.domain_registry.domains.get(domain_name)
        
        if not domain_obj:
            return {}
        
        # Handle dict (fallback stub) vs DomainProfile (real implementation)
        if isinstance(domain_obj, dict):
            return domain_obj.get('characteristics', {})
        else:
            # DomainProfile object - access characteristics attribute
            return getattr(domain_obj, 'characteristics', {})
    
    def _calculate_domain_similarity(self, source_domain: str, target_domain: str) -> float:
        """
        Calculate similarity between domains (FIXED: DomainProfile compatibility)
        
        Args:
            source_domain: Source domain
            target_domain: Target domain
            
        Returns:
            Similarity score (0-1)
        """
        if source_domain == target_domain:
            return 1.0
        
        # Get domain characteristics using helper method
        source_chars = self._get_domain_characteristics(source_domain)
        target_chars = self._get_domain_characteristics(target_domain)
        
        if not source_chars or not target_chars:
            return 0.5  # Unknown similarity
        
        # Compare characteristics
        similarity = 0.0
        comparison_count = 0
        
        # Compare adaptability
        if 'adaptability' in source_chars and 'adaptability' in target_chars:
            adaptability_levels = {'low': 0, 'medium': 1, 'high': 2}
            source_level = adaptability_levels.get(source_chars['adaptability'], 1)
            target_level = adaptability_levels.get(target_chars['adaptability'], 1)
            similarity += 1.0 - abs(source_level - target_level) / 2.0
            comparison_count += 1
        
        # Compare complexity
        if 'complexity' in source_chars and 'complexity' in target_chars:
            complexity_levels = {'low': 0, 'medium': 1, 'high': 2, 'very_high': 3}
            source_level = complexity_levels.get(source_chars['complexity'], 1)
            target_level = complexity_levels.get(target_chars['complexity'], 1)
            similarity += 1.0 - abs(source_level - target_level) / 3.0
            comparison_count += 1
        
        if comparison_count > 0:
            return similarity / comparison_count
        
        return 0.5
    
    def validate_transfer_compatibility(self, concept: Concept, source: str, 
                                       target: str) -> TransferCompatibility:
        """
        Validate if concept can transfer between domains - FIXED to use real TransferEngine API
        
        Args:
            concept: Concept to transfer
            source: Source domain
            target: Target domain
            
        Returns:
            Transfer compatibility assessment
        """
        # FIXED: Use the actual TransferEngine API methods
        # First try full transfer validation
        transfer_decision = self.transfer_engine.validate_full_transfer(concept, source, target)
        
        # Convert TransferDecision to TransferCompatibility
        concept_id = getattr(concept, 'concept_id', str(concept))
        
        # Calculate compatibility score from decision
        compatibility_score = transfer_decision.confidence
        
        # Extract required adaptations from mitigations
        required_adaptations = []
        if hasattr(transfer_decision, 'mitigations'):
            for mitigation in transfer_decision.mitigations:
                if hasattr(mitigation, 'description'):
                    required_adaptations.append(mitigation.description)
                elif hasattr(mitigation, 'mitigation_id'):
                    required_adaptations.append(mitigation.mitigation_id)
        
        # Extract risks from risk assessment and reasoning
        risks = []
        if hasattr(transfer_decision, 'risk_assessment'):
            for risk_type, risk_value in transfer_decision.risk_assessment.items():
                if risk_value > 0.5:  # High risk
                    risks.append(f"{risk_type}: {risk_value:.2f}")
        
        if hasattr(transfer_decision, 'reasoning'):
            for reason in transfer_decision.reasoning:
                if 'risk' in reason.lower() or 'blocked' in reason.lower():
                    risks.append(reason)
        
        # Create compatibility object
        compatibility = TransferCompatibility(
            source_domain=source,
            target_domain=target,
            concept_id=concept_id,
            compatibility_score=compatibility_score,
            required_adaptations=required_adaptations,
            risks=risks,
            confidence=transfer_decision.confidence
        )
        
        # SAFETY: Validate transfer with safety validator
        if self.safety_validator:
            # Check if domains are in safe regions
            source_safe = self.safety_validator.is_safe_region({'domain': source})
            target_safe = self.safety_validator.is_safe_region({'domain': target})
            
            if not source_safe or not target_safe:
                logger.warning("Transfer involves unsafe domain region")
                compatibility.risks.append("unsafe_domain_region")
                compatibility.compatibility_score *= 0.5
        
        # Additional validation based on domain registry (FIXED: DomainProfile compatibility)
        source_chars = self._get_domain_characteristics(source)
        target_chars = self._get_domain_characteristics(target)
        
        # Check complexity compatibility
        if source_chars.get('complexity') == 'low' and target_chars.get('complexity') == 'very_high':
            compatibility.risks.append("significant_complexity_increase")
            compatibility.compatibility_score *= 0.7
        
        # Check adaptability
        if target_chars.get('adaptability') == 'low':
            compatibility.required_adaptations.append("rigid_domain_constraints")
        
        logger.debug("Validated transfer compatibility for concept: %.2f",
                    compatibility.compatibility_score)
        
        return compatibility
    
    def resolve_concept_conflict(self, new_pattern, existing_concepts: List[Concept]) -> Dict[str, Any]:
        """
        Resolve conflict between new pattern and existing concepts
        
        Args:
            new_pattern: New pattern
            existing_concepts: Existing concepts that conflict
            
        Returns:
            Resolution decision
        """
        # Create temporary concept from new pattern
        new_concept = self.concept_mapper.map_pattern_to_concept(new_pattern)
        
        resolutions = []
        
        for existing_concept in existing_concepts:
            # Detect conflict type and severity
            conflict = self._detect_conflict(new_concept, existing_concept)
            
            if conflict:
                # Resolve conflict
                resolution = self.conflict_resolver.resolve_conflict(conflict)
                resolutions.append(resolution)
                
                # Apply resolution
                self._apply_resolution(resolution, new_concept, existing_concept)
        
        self.total_conflicts += len(resolutions)
        
        # Aggregate resolutions
        final_resolution = self._aggregate_resolutions(resolutions)
        
        logger.info("Resolved %d conflicts for new pattern", len(resolutions))
        
        return final_resolution
    
    def get_applicable_concepts(self, domain: str, min_confidence: float = 0.7) -> List[Concept]:
        """
        Get concepts applicable to domain using inverted index (FIXED: optimized with inverted index)
        
        Args:
            domain: Target domain
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of applicable concepts
        """
        # FIXED: Rebuild index if dirty
        if self.index_dirty:
            self._rebuild_domain_concept_index()
        
        # Clear cache if TTL expired
        if time.time() - self.last_cache_clear > self.cache_ttl:
            self.domain_concept_cache.clear()
            self.last_cache_clear = time.time()
        
        # Check cache
        cache_key = f"{domain}_{min_confidence}"
        if cache_key in self.domain_concept_cache:
            cached_concepts, cached_time = self.domain_concept_cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                logger.debug("Using cached concepts for domain %s", domain)
                self.cache_manager.record_hit('domain_concept')
                return cached_concepts
        
        self.cache_manager.record_miss('domain_concept')
        
        # FIXED: Use inverted index for fast lookup
        with self._concept_lock:
            concept_ids = self.domain_concept_index.get(domain, set())
            
            applicable = []
            
            for concept_id in concept_ids:
                concept = self.concept_mapper.concepts.get(concept_id)
                if not concept:
                    continue
                
                # Check confidence threshold
                if hasattr(concept, 'confidence') and concept.confidence >= min_confidence:
                    applicable.append(concept)
            
            # Also check concepts that can transfer (but limit this expensive operation)
            transfer_candidates = []
            all_concepts = list(self.concept_mapper.concepts.values())
            
            # Only check transfer for top concepts by confidence
            all_concepts.sort(key=lambda c: getattr(c, 'confidence', 0), reverse=True)
            
            for concept in all_concepts[:50]:  # Limit to top 50
                if hasattr(concept, 'domains') and domain not in concept.domains:
                    if hasattr(concept, 'confidence') and concept.confidence >= min_confidence - 0.1:
                        # Check transfer compatibility
                        for source_domain in concept.domains:
                            compatibility = self.validate_transfer_compatibility(
                                concept, source_domain, domain
                            )
                            
                            if compatibility.is_compatible() and compatibility.confidence >= min_confidence:
                                # FIXED: Use actual transfer_engine method
                                # transfer_concept doesn't exist in real TransferEngine, so we execute transfer properly
                                transfer_decision = self.transfer_engine.validate_full_transfer(
                                    concept, source_domain, domain
                                )
                                
                                if hasattr(transfer_decision, 'is_transferable') and transfer_decision.is_transferable():
                                    result = self.transfer_engine.execute_transfer(
                                        concept, transfer_decision, domain
                                    )
                                    
                                    if result.get('success'):
                                        transferred = result.get('transferred_concept')
                                        if transferred:
                                            transfer_candidates.append(transferred)
                                            self.total_transfers += 1
                                            break
            
            applicable.extend(transfer_candidates)
            
            # Sort by combined metric
            def sort_key(c):
                conf = getattr(c, 'confidence', 0.5)
                succ = getattr(c, 'success_rate', 0.5)
                return conf * succ
            
            applicable.sort(key=sort_key, reverse=True)
            
            # Update domain registry
            for concept in applicable:
                if hasattr(concept, 'concept_id'):
                    if domain not in self.domain_registry.domains:
                        self.domain_registry.register_domain(domain)
                    
                    domain_obj = self.domain_registry.domains.get(domain)
                    if domain_obj:
                        # Handle both dict and DomainProfile
                        if isinstance(domain_obj, dict):
                            if 'concepts' not in domain_obj:
                                domain_obj['concepts'] = set()
                            domain_obj['concepts'].add(concept.concept_id)
                        else:
                            # DomainProfile object
                            if not hasattr(domain_obj, 'concepts'):
                                domain_obj.concepts = set()
                            domain_obj.concepts.add(concept.concept_id)
            
            # FIXED: Cache with size limit
            if len(self.domain_concept_cache) < self.max_domain_cache_size:
                self.domain_concept_cache[cache_key] = (applicable, time.time())
            else:
                # Evict oldest entry
                oldest_key = min(self.domain_concept_cache.keys(),
                               key=lambda k: self.domain_concept_cache[k][1])
                del self.domain_concept_cache[oldest_key]
                self.domain_concept_cache[cache_key] = (applicable, time.time())
            
            logger.debug("Returning %d applicable concepts for domain %s", len(applicable), domain)
            
            return applicable
    
    def _get_concepts_by_domain(self, domain: str) -> List[Concept]:
        """
        Fast domain-based filtering
        
        Args:
            domain: Target domain
            
        Returns:
            Concepts that have this domain
        """
        candidates = []
        
        for concept in self.concept_mapper.concepts.values():
            if hasattr(concept, 'domains'):
                if isinstance(concept.domains, set):
                    if domain in concept.domains:
                        candidates.append(concept)
                elif isinstance(concept.domains, list):
                    if domain in concept.domains:
                        candidates.append(concept)
        
        return candidates
    
    def _create_concept_version(self, concept: Concept, changes: List[str]):
        """
        Create new version of concept (FIXED: version size limits)
        """
        concept_id = getattr(concept, 'concept_id', str(concept))
        
        # FIXED: Enforce versioned concepts limit
        if concept_id not in self.concept_versions:
            if len(self.concept_versions) >= self.max_versioned_concepts:
                self._evict_oldest_versioned_concept()
            self.concept_versions[concept_id] = []
        
        # FIXED: Limit versions per concept
        if len(self.concept_versions[concept_id]) >= self.max_versions_per_concept:
            # Remove oldest version
            self.concept_versions[concept_id].pop(0)
        
        version_num = len(self.concept_versions[concept_id]) + 1
        version = ConceptVersion(
            version_id=f"{concept_id}_v{version_num}",
            concept_id=concept_id,
            version_number=version_num,
            changes=changes,
            performance={'success_rate': getattr(concept, 'success_rate', 0.5)}
        )
        
        if self.concept_versions[concept_id]:
            version.parent_version = self.concept_versions[concept_id][-1].version_id
        
        self.concept_versions[concept_id].append(version)
        
        # FIXED: Enforce current versions limit
        if len(self.current_versions) >= self.max_current_versions:
            # Remove random entry (could use LRU)
            oldest_key = next(iter(self.current_versions))
            del self.current_versions[oldest_key]
        
        self.current_versions[concept_id] = version.version_id
    
    def _detect_conflict(self, new_concept: Concept, existing_concept: Concept) -> Optional[ConceptConflict]:
        """Detect conflict between concepts"""
        # Calculate similarity
        similar_concepts = self.concept_mapper.find_similar_concepts(new_concept, top_k=10)
        
        for similar, similarity in similar_concepts:
            similar_id = getattr(similar, 'concept_id', str(similar))
            existing_id = getattr(existing_concept, 'concept_id', str(existing_concept))
            
            if similar_id == existing_id and similarity > 0.7:
                # High similarity indicates potential conflict
                conflict_type = self._determine_conflict_type(new_concept, existing_concept)
                
                return ConceptConflict(
                    new_concept=new_concept,
                    existing_concept=existing_concept,
                    conflict_type=conflict_type,
                    severity=similarity,
                    resolution_options=['replace', 'merge', 'coexist', 'reject']
                )
        
        return None
    
    def _determine_conflict_type(self, concept1: Concept, concept2: Concept) -> str:
        """Determine type of conflict"""
        c1_id = getattr(concept1, 'concept_id', '')
        c2_id = getattr(concept2, 'concept_id', '')
        
        if hasattr(concept1, 'name') and hasattr(concept2, 'name'):
            if concept1.name == concept2.name:
                return "name_conflict"
        
        if hasattr(concept1, 'get_signature') and hasattr(concept2, 'get_signature'):
            if concept1.get_signature() == concept2.get_signature():
                return "duplicate"
        
        if hasattr(concept1, 'features') and hasattr(concept2, 'features'):
            if len(set(concept1.features.keys()) & set(concept2.features.keys())) > 5:
                return "feature_overlap"
        
        return "semantic_similarity"
    
    def _apply_resolution(self, resolution: Dict[str, Any], new_concept: Concept, 
                         existing_concept: Concept):
        """Apply conflict resolution"""
        action = resolution.get('action', 'coexist')
        
        with self._concept_lock:
            if action == 'replace':
                # Replace existing with new
                existing_id = getattr(existing_concept, 'concept_id', str(existing_concept))
                self.concept_mapper.concepts[existing_id] = new_concept
                new_concept.concept_id = existing_id  # Keep same ID
                self._create_concept_version(new_concept, ["Replaced existing concept"])
                self.index_dirty = True  # Mark index as needing rebuild
                
            elif action == 'merge':
                # Merge concepts
                if hasattr(existing_concept, 'features') and hasattr(new_concept, 'features'):
                    existing_concept.features.update(new_concept.features)
                if hasattr(existing_concept, 'domains') and hasattr(new_concept, 'domains'):
                    existing_concept.domains.update(new_concept.domains)
                if hasattr(existing_concept, 'confidence') and hasattr(new_concept, 'confidence'):
                    existing_concept.confidence = (existing_concept.confidence + new_concept.confidence) / 2
                self._create_concept_version(existing_concept, ["Merged with new concept"])
                self.index_dirty = True  # Mark index as needing rebuild
                
            elif action == 'coexist':
                # Register new concept with different ID
                if hasattr(new_concept, 'concept_id'):
                    new_concept.concept_id = f"{new_concept.concept_id}_alt"
                self.concept_mapper.register_concept(new_concept)
                self.index_dirty = True  # Mark index as needing rebuild
                
            # 'reject' action requires no changes
    
    def _aggregate_resolutions(self, resolutions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate multiple resolutions"""
        if not resolutions:
            return {'action': 'none', 'confidence': 1.0}
        
        # Count actions
        action_counts = defaultdict(int)
        total_confidence = 0.0
        
        for resolution in resolutions:
            action_counts[resolution.get('action', 'none')] += 1
            total_confidence += resolution.get('confidence', 0.5)
        
        # Most common action wins
        final_action = max(action_counts, key=action_counts.get)
        avg_confidence = total_confidence / len(resolutions)
        
        return {
            'action': final_action,
            'confidence': avg_confidence,
            'resolution_count': len(resolutions),
            'action_distribution': dict(action_counts)
        }
    
    def _initialize_default_domains(self):
        """Initialize with default domains"""
        default_domains = {
            'general': {'adaptability': 'high', 'complexity': 'medium'},
            'optimization': {'adaptability': 'medium', 'complexity': 'high'},
            'classification': {'adaptability': 'high', 'complexity': 'medium'},
            'generation': {'adaptability': 'medium', 'complexity': 'high'},
            'planning': {'adaptability': 'low', 'complexity': 'very_high'},
            'control': {'adaptability': 'low', 'complexity': 'high'},
            'reasoning': {'adaptability': 'medium', 'complexity': 'very_high'},
            'perception': {'adaptability': 'medium', 'complexity': 'medium'}
        }
        
        for name, chars in default_domains.items():
            self.domain_registry.register_domain(name, characteristics=chars)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get bridge statistics"""
        stats = {
            'total_concepts': self.total_concepts,
            'active_concepts': len(self.concept_mapper.concepts),
            'total_transfers': self.total_transfers,
            'total_conflicts': self.total_conflicts,
            'domains': len(self.domain_registry.domains),
            'versioned_concepts': len(self.concept_versions),
            'total_versions': sum(len(v) for v in self.concept_versions.values()),
            'current_versions_tracked': len(self.current_versions),
            'pattern_cache_size': len(self.pattern_signature_cache),
            'domain_cache_size': len(self.domain_concept_cache),
            'operations_logged': len(self.operation_history),
            'world_model_connected': self.world_model is not None,
            'index_domains': len(self.domain_concept_index),
            'index_total_concepts': sum(len(s) for s in self.domain_concept_index.values()),
            'cache_manager': self.cache_manager.get_statistics(),
            'max_versioned_concepts': self.max_versioned_concepts,
            'max_pattern_cache_size': self.max_pattern_cache_size,
            'max_domain_cache_size': self.max_domain_cache_size,
            'max_index_domains': self.max_index_domains
        }
        
        # Add safety statistics
        if self.safety_validator:
            stats['safety'] = {
                'enabled': True,
                'blocks': dict(self.safety_blocks),
                'corrections': dict(self.safety_corrections),
                'total_blocks': sum(self.safety_blocks.values()),
                'total_corrections': sum(self.safety_corrections.values())
            }
        else:
            stats['safety'] = {'enabled': False}
        
        return stats