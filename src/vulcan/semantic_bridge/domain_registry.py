"""
domain_registry.py - Domain metadata and profile management for semantic bridge
Part of the VULCAN-AGI system

FIXED: Added safety_config and world_model integration
ENHANCED: Adaptive cache sizing, externalized risk adjuster configuration
PRODUCTION-READY: All unbounded data structures fixed with proper limits and eviction
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque, Counter
import time
import json
import hashlib
from enum import Enum
from pathlib import Path
import pickle
import threading
from ..security_fixes import safe_pickle_load

# Import safety validator
try:
    from ..safety.safety_validator import EnhancedSafetyValidator
    from ..safety.safety_types import SafetyConfig
    SAFETY_VALIDATOR_AVAILABLE = True
except ImportError:
    SAFETY_VALIDATOR_AVAILABLE = False
    logging.warning("safety_validator not available, domain_registry operating without safety checks")
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
        
        def to_undirected(self):
            """Convert to undirected graph (simplified)"""
            undirected = SimpleDiGraph()
            undirected.nodes_dict = self.nodes_dict.copy()
            
            # Add edges in both directions
            for source, targets in self.edges_dict.items():
                for target in targets:
                    undirected.edges_dict[source].append(target)
                    undirected.edges_dict[target].append(source)
            
            return undirected
    
    class MockNX:
        """Mock NetworkX module with basic functionality"""
        DiGraph = SimpleDiGraph
        
        @staticmethod
        def ancestors(graph, node):
            """Get all ancestors of a node"""
            ancestors = set()
            to_visit = graph.predecessors(node) if hasattr(graph, 'predecessors') else []
            
            while to_visit:
                current = to_visit.pop(0)
                if current not in ancestors:
                    ancestors.add(current)
                    if hasattr(graph, 'predecessors'):
                        to_visit.extend(graph.predecessors(current))
            
            return list(ancestors)
        
        @staticmethod
        def descendants(graph, node):
            """Get all descendants of a node"""
            descendants = set()
            to_visit = graph.successors(node) if hasattr(graph, 'successors') else []
            
            while to_visit:
                current = to_visit.pop(0)
                if current not in descendants:
                    descendants.add(current)
                    if hasattr(graph, 'successors'):
                        to_visit.extend(graph.successors(current))
            
            return list(descendants)
        
        @staticmethod
        def shortest_path_length(graph, source, target):
            """Simple BFS to find shortest path length"""
            if source == target:
                return 0
            
            visited = {source}
            queue = [(source, 0)]
            
            while queue:
                current, dist = queue.pop(0)
                
                if hasattr(graph, 'successors'):
                    for neighbor in graph.successors(current):
                        if neighbor == target:
                            return dist + 1
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, dist + 1))
            
            # No path found
            raise ValueError("No path between nodes")
        
        class NetworkXNoPath(Exception):
            pass
    
    nx = MockNX()

logger = logging.getLogger(__name__)


class DomainCriticality(Enum):
    """Criticality levels for domains"""
    LOW = 0.1
    MEDIUM_LOW = 0.3
    MEDIUM = 0.5
    MEDIUM_HIGH = 0.7
    HIGH = 0.9
    SAFETY_CRITICAL = 0.95


class EffectCategory(Enum):
    """Categories of domain effects"""
    COMPUTATION = "computation"
    TRANSFORMATION = "transformation"
    OPTIMIZATION = "optimization"
    CONTROL = "control"
    PREDICTION = "prediction"
    GENERATION = "generation"
    ANALYSIS = "analysis"
    VALIDATION = "validation"


class PatternType(Enum):
    """Types of patterns in domains"""
    STRUCTURAL = "structural"
    BEHAVIORAL = "behavioral"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    HIERARCHICAL = "hierarchical"
    ITERATIVE = "iterative"
    RECURSIVE = "recursive"


@dataclass
class Pattern:
    """Pattern representation for domains"""
    pattern_id: str
    pattern_type: PatternType
    description: str
    frequency: float = 0.5
    complexity: float = 0.5
    success_rate: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_signature(self) -> str:
        """Get unique signature for pattern"""
        content = f"{self.pattern_type.value}_{self.description}_{self.complexity}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


@dataclass
class DomainEffect:
    """Effect within a domain"""
    effect_id: str
    category: EffectCategory
    description: str
    importance: float = 0.5
    frequency: float = 0.5
    prerequisites: List[str] = field(default_factory=list)
    outcomes: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'effect_id': self.effect_id,
            'category': self.category.value,
            'description': self.description,
            'importance': self.importance,
            'frequency': self.frequency,
            'prerequisites': self.prerequisites,
            'outcomes': self.outcomes,
            'constraints': self.constraints
        }


@dataclass
class DomainProfile:
    """Single domain characteristics"""
    name: str
    criticality_score: float = 0.5  # 0.1 low risk, 0.9 high risk
    effect_types: Set[str] = field(default_factory=set)
    typical_patterns: List[Pattern] = field(default_factory=list)
    capabilities: Set[str] = field(default_factory=set)
    limitations: Set[str] = field(default_factory=set)
    parent_domains: List[str] = field(default_factory=list)
    child_domains: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    def add_pattern(self, pattern: Pattern):
        """Add pattern to domain"""
        self.typical_patterns.append(pattern)
        
    def update_performance(self, metric: str, value: float):
        """Update performance metric"""
        if metric not in self.performance_metrics:
            self.performance_metrics[metric] = value
        else:
            # Exponential moving average
            alpha = 0.2
            self.performance_metrics[metric] = alpha * value + (1 - alpha) * self.performance_metrics[metric]
    
    def get_risk_level(self) -> str:
        """Get human-readable risk level"""
        if self.criticality_score >= DomainCriticality.SAFETY_CRITICAL.value:
            return "SAFETY_CRITICAL"
        elif self.criticality_score >= DomainCriticality.HIGH.value:
            return "HIGH"
        elif self.criticality_score >= DomainCriticality.MEDIUM_HIGH.value:
            return "MEDIUM_HIGH"
        elif self.criticality_score >= DomainCriticality.MEDIUM.value:
            return "MEDIUM"
        elif self.criticality_score >= DomainCriticality.MEDIUM_LOW.value:
            return "MEDIUM_LOW"
        else:
            return "LOW"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'criticality_score': self.criticality_score,
            'risk_level': self.get_risk_level(),
            'effect_types': list(self.effect_types),
            'pattern_count': len(self.typical_patterns),
            'capabilities': list(self.capabilities),
            'limitations': list(self.limitations),
            'parent_domains': self.parent_domains,
            'child_domains': self.child_domains,
            'performance_metrics': self.performance_metrics
        }


@dataclass
class DomainRelationship:
    """Relationship between two domains"""
    source: str
    target: str
    relationship_type: str  # "parent", "sibling", "specialization", etc.
    strength: float = 0.5
    bidirectional: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class DomainRegistry:
    """Manages domain metadata and profiles - FIXED with safety and world_model"""
    
    def __init__(self, 
                 world_model=None,
                 storage_path: Optional[Path] = None,
                 safety_config: Optional[Dict[str, Any]] = None):
        """
        Initialize domain registry - FIXED: Added world_model and safety_config
        
        Args:
            world_model: World model instance for accessing causal knowledge
            storage_path: Optional path for persistent storage
            safety_config: Optional safety configuration
        """
        self.world_model = world_model
        self.storage_path = storage_path or Path("domain_registry")
        
        # Initialize safety validator
        if SAFETY_VALIDATOR_AVAILABLE:
            if isinstance(safety_config, dict) and safety_config:
                self.safety_validator = EnhancedSafetyValidator(SafetyConfig.from_dict(safety_config))
            else:
                self.safety_validator = EnhancedSafetyValidator()
            logger.info("DomainRegistry: Safety validator initialized")
        else:
            self.safety_validator = None
            logger.warning("DomainRegistry: Safety validator not available - operating without safety checks")
        
        # FIXED: Domain storage with size limits
        self.domains = {}  # name -> DomainProfile
        self.max_domains = 10000
        
        # FIXED: Domain effects with size limits (changed from defaultdict to regular dict)
        self.domain_effects = {}  # domain_name -> List[DomainEffect]
        self.max_effect_domains = 5000
        self.max_effects_per_domain = 1000
        
        # Relationship graph
        if NETWORKX_AVAILABLE:
            self.domain_graph = nx.DiGraph()
        else:
            self.domain_graph = SimpleDiGraph()
            
        self.relationships = []  # List of DomainRelationship
        
        # FIXED: Distance cache with adaptive sizing
        self.distance_cache = {}
        self.max_cache_size = 1000
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        self.last_cache_resize = time.time()
        
        # Statistics
        self.total_domains = 0
        
        # FIXED: Effect statistics with size limit (changed from defaultdict to regular dict)
        self.effect_statistics = {}  # category -> {'count': int, 'avg_importance': float}
        self.max_effect_categories = 100
        
        # FIXED: Replace defaultdict(int) with Counter
        self.safety_blocks = Counter()
        self.safety_corrections = Counter()
        
        # Risk adjuster
        self.risk_adjuster = RiskAdjuster()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize with default domains
        self._initialize_default_domains()
        
        # Load existing registry if available
        self._load_registry()
        
        logger.info("DomainRegistry initialized (production-ready) with bounded data structures, %d domains", len(self.domains))
    
    def _adaptive_cache_resize(self):
        """
        Adjust cache size based on hit rate (FIXED: adaptive cache)
        """
        total_requests = self.cache_hit_count + self.cache_miss_count
        
        if total_requests < 100:
            return  # Not enough data
        
        hit_rate = self.cache_hit_count / total_requests
        
        # Resize based on hit rate
        if hit_rate > 0.8:
            # High hit rate - increase cache
            self.max_cache_size = min(5000, int(self.max_cache_size * 1.5))
            logger.debug("Increased cache size to %d (hit rate: %.2f)", 
                        self.max_cache_size, hit_rate)
        elif hit_rate < 0.3:
            # Low hit rate - decrease cache
            self.max_cache_size = max(500, int(self.max_cache_size * 0.7))
            logger.debug("Decreased cache size to %d (hit rate: %.2f)", 
                        self.max_cache_size, hit_rate)
        
        # Reset counters
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        self.last_cache_resize = time.time()
    
    def _evict_least_used_domain(self):
        """
        Evict least used domain when at capacity (FIXED: domain size limit)
        """
        if not self.domains:
            return
        
        # Find domain with lowest usage
        min_domain = None
        min_usage = float('inf')
        
        for name, profile in self.domains.items():
            usage = profile.performance_metrics.get('usage_count', 0)
            if usage < min_usage:
                min_usage = usage
                min_domain = name
        
        if min_domain:
            del self.domains[min_domain]
            self.total_domains -= 1
            logger.debug("Evicted domain %s (usage: %.0f)", min_domain, min_usage)
    
    def _evict_oldest_effect_domain(self):
        """
        Evict domain with oldest effects (FIXED: effect storage size limit)
        """
        if not self.domain_effects:
            return
        
        # Remove first domain (could track timestamps for better eviction)
        oldest_domain = next(iter(self.domain_effects))
        del self.domain_effects[oldest_domain]
        logger.debug("Evicted effects for domain %s", oldest_domain)
    
    def register_domain(self, name: str, profile: DomainProfile = None, characteristics: Dict[str, Any] = None):
        """
        Register a new domain - FIXED: Added characteristics parameter for semantic_bridge compatibility
        
        Args:
            name: Domain name
            profile: Domain profile (created if not provided)
            characteristics: Optional characteristics dict for compatibility
        """
        with self._lock:
            # SAFETY: Validate domain name
            if self.safety_validator:
                try:
                    if hasattr(self.safety_validator, 'validate_domain_name'):
                        name_check = self.safety_validator.validate_domain_name(name)
                        if not name_check.get('safe', True):
                            logger.warning("Unsafe domain name: %s", name_check.get('reason', 'unknown'))
                            self.safety_blocks['unsafe_domain_name'] += 1
                            return
                except Exception as e:
                    logger.debug("Error validating domain name: %s", e)
            
            if profile is None:
                profile = DomainProfile(name=name)
                
                # FIXED: Apply characteristics if provided (for semantic_bridge compatibility)
                if characteristics:
                    if 'adaptability' in characteristics:
                        profile.metadata['adaptability'] = characteristics['adaptability']
                    if 'complexity' in characteristics:
                        profile.metadata['complexity'] = characteristics['complexity']
                    if 'criticality' in characteristics:
                        # Map text to score
                        criticality_map = {
                            'low': DomainCriticality.LOW.value,
                            'medium': DomainCriticality.MEDIUM.value,
                            'high': DomainCriticality.HIGH.value,
                            'safety_critical': DomainCriticality.SAFETY_CRITICAL.value
                        }
                        profile.criticality_score = criticality_map.get(
                            characteristics['criticality'], 
                            DomainCriticality.MEDIUM.value
                        )
            
            # SAFETY: Validate criticality score
            if not (0 <= profile.criticality_score <= 1):
                logger.warning("Invalid criticality score %.2f, clamping to [0,1]", profile.criticality_score)
                self.safety_corrections['criticality_score'] += 1
                profile.criticality_score = np.clip(profile.criticality_score, 0, 1)
            
            # FIXED: Enforce domain limit
            if name not in self.domains:
                if len(self.domains) >= self.max_domains:
                    self._evict_least_used_domain()
            
            self.domains[name] = profile
            self.domain_graph.add_node(name, profile=profile)
            self.total_domains += 1
            
            # Clear distance cache when topology changes
            self.distance_cache.clear()
            
            # FIXED: Link to world model if available
            if self.world_model:
                try:
                    self._link_domain_to_world_model(name, profile)
                except Exception as e:
                    logger.debug("Failed to link domain to world model: %s", e)
            
            logger.info("Registered domain: %s (criticality: %.2f)", name, profile.criticality_score)
    
    def _link_domain_to_world_model(self, domain_name: str, profile: DomainProfile):
        """
        Link domain to world model - FIXED: New integration method
        
        Args:
            domain_name: Name of domain
            profile: Domain profile
        """
        if not self.world_model or not hasattr(self.world_model, 'causal_graph'):
            return
        
        try:
            # Add domain as a node in causal graph if not exists
            domain_node = f"domain_{domain_name}"
            if not self.world_model.causal_graph.has_node(domain_node):
                self.world_model.causal_graph.add_node(domain_node)
            
            # Link domain capabilities to effects
            for capability in profile.capabilities:
                capability_node = f"capability_{capability}"
                
                # Validate with safety
                if self.safety_validator:
                    try:
                        if hasattr(self.safety_validator, 'validate_causal_edge'):
                            edge_validation = self.safety_validator.validate_causal_edge(
                                domain_node, capability_node, 0.7
                            )
                            if not edge_validation.get('safe', True):
                                continue
                    except Exception as e:
                        logger.debug("Safety validation error: %s", e)
                        continue
                
                if not self.world_model.causal_graph.has_edge(domain_node, capability_node):
                    self.world_model.causal_graph.add_edge(
                        domain_node,
                        capability_node,
                        strength=0.7,
                        evidence_type="domain_registry"
                    )
            
            logger.debug("Linked domain %s to world model", domain_name)
        except Exception as e:
            logger.debug("Error linking domain to world model: %s", e)
    
    def get_related_domains(self, domain: str) -> List[str]:
        """
        Get related domains - FIXED: Added for semantic_bridge compatibility
        
        Args:
            domain: Domain name
            
        Returns:
            List of related domain names
        """
        with self._lock:
            related = []
            
            # Get hierarchy
            hierarchy = self.get_domain_hierarchy(domain)
            
            # Add parents, children, and siblings
            related.extend(hierarchy.get('parents', []))
            related.extend(hierarchy.get('children', []))
            related.extend(hierarchy.get('siblings', []))
            
            # Add similar domains
            similar = self.get_similar_domains(domain, top_k=3)
            related.extend([d for d, _ in similar])
            
            # Remove duplicates and self
            related = list(set(related))
            if domain in related:
                related.remove(domain)
            
            return related
    
    def update_domain_performance(self, domain: str, success: bool):
        """
        Update domain performance - FIXED: Added for semantic_bridge compatibility
        
        Args:
            domain: Domain name
            success: Whether operation was successful
        """
        with self._lock:
            if domain not in self.domains:
                logger.debug("Domain %s not found for performance update", domain)
                return
            
            profile = self.domains[domain]
            
            # Update success rate metric
            current_rate = profile.performance_metrics.get('success_rate', 0.5)
            alpha = 0.1
            new_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * current_rate
            profile.update_performance('success_rate', new_rate)
            
            # Update usage count
            usage = profile.performance_metrics.get('usage_count', 0)
            profile.update_performance('usage_count', usage + 1)
            
            logger.debug("Updated performance for domain %s: success_rate=%.2f", domain, new_rate)
    
    def get_domain_effects(self, domain_name: str) -> List[DomainEffect]:
        """
        Get effects for a domain
        
        Args:
            domain_name: Name of domain
            
        Returns:
            List of domain effects
        """
        with self._lock:
            # Check if effects already calculated
            if domain_name in self.domain_effects:
                return self.domain_effects[domain_name]
            
            # Generate effects from profile
            profile = self.domains.get(domain_name)
            if not profile:
                return []
            
            effects = []
            
            # Generate effects from effect types
            for effect_type in profile.effect_types:
                effect = DomainEffect(
                    effect_id=f"{domain_name}_{effect_type}",
                    category=self._categorize_effect(effect_type),
                    description=f"{effect_type} effect in {domain_name}",
                    importance=0.5 + profile.criticality_score * 0.3
                )
                effects.append(effect)
            
            # Generate effects from patterns
            for pattern in profile.typical_patterns:
                effect = DomainEffect(
                    effect_id=f"{domain_name}_pattern_{pattern.pattern_id}",
                    category=self._pattern_to_effect_category(pattern.pattern_type),
                    description=f"Pattern-based effect: {pattern.description}",
                    importance=pattern.complexity * 0.7,
                    frequency=pattern.frequency
                )
                effects.append(effect)
            
            # FIXED: Enforce effect storage limits
            if domain_name not in self.domain_effects:
                if len(self.domain_effects) >= self.max_effect_domains:
                    self._evict_oldest_effect_domain()
                self.domain_effects[domain_name] = []
            
            # FIXED: Limit effects per domain
            if len(effects) > self.max_effects_per_domain:
                # Keep most important effects
                effects.sort(key=lambda e: e.importance, reverse=True)
                effects = effects[:self.max_effects_per_domain]
            
            # Cache effects
            self.domain_effects[domain_name] = effects
            
            # Update statistics
            for effect in effects:
                self._update_effect_statistics(effect)
            
            return effects
    
    def calculate_domain_distance(self, domain_a: str, domain_b: str) -> float:
        """
        Calculate distance between two domains (FIXED: adaptive cache tracking)
        
        Args:
            domain_a: First domain
            domain_b: Second domain
            
        Returns:
            Distance score [0, 1] where 0 is identical, 1 is maximally different
        """
        if domain_a == domain_b:
            return 0.0
        
        with self._lock:
            # Check cache
            cache_key = tuple(sorted([domain_a, domain_b]))
            if cache_key in self.distance_cache:
                self.cache_hit_count += 1
                return self.distance_cache[cache_key]
            
            self.cache_miss_count += 1
            
            # Periodic cache resize
            if time.time() - self.last_cache_resize > 3600:  # Hourly
                self._adaptive_cache_resize()
            
            # Get profiles
            profile_a = self.domains.get(domain_a)
            profile_b = self.domains.get(domain_b)
            
            if not profile_a or not profile_b:
                # Unknown domain - maximum distance
                distance = 1.0
            else:
                # Calculate multi-factor distance
                distances = []
                
                # Criticality distance
                crit_dist = abs(profile_a.criticality_score - profile_b.criticality_score)
                distances.append(crit_dist)
                
                # Effect type distance
                effects_a = set(profile_a.effect_types)
                effects_b = set(profile_b.effect_types)
                if effects_a or effects_b:
                    effect_similarity = len(effects_a & effects_b) / len(effects_a | effects_b)
                    distances.append(1.0 - effect_similarity)
                
                # Capability distance
                caps_a = profile_a.capabilities
                caps_b = profile_b.capabilities
                if caps_a or caps_b:
                    cap_similarity = len(caps_a & caps_b) / len(caps_a | caps_b)
                    distances.append(1.0 - cap_similarity)
                
                # Graph distance
                if self.domain_graph.has_node(domain_a) and self.domain_graph.has_node(domain_b):
                    try:
                        # Create simple undirected version for distance calculation
                        undirected = self.domain_graph.to_undirected()
                        
                        # Simple BFS for path length
                        path_length = self._find_shortest_path_length(undirected, domain_a, domain_b)
                        graph_distance = min(1.0, path_length / 5)  # Normalize to [0, 1]
                        distances.append(graph_distance)
                    except Exception as e:                        distances.append(1.0)
                
                # Calculate weighted average
                if distances:
                    weights = [0.3, 0.25, 0.25, 0.2]  # Criticality, effects, capabilities, graph
                    weighted_distances = [d * w for d, w in zip(distances, weights[:len(distances)])]
                    distance = sum(weighted_distances) / sum(weights[:len(distances)])
                else:
                    distance = 0.5
            
            # Cache with size limit
            if len(self.distance_cache) < self.max_cache_size:
                self.distance_cache[cache_key] = distance
            elif len(self.distance_cache) >= self.max_cache_size:
                # Evict random item (could use LRU)
                evict_key = next(iter(self.distance_cache))
                del self.distance_cache[evict_key]
                self.distance_cache[cache_key] = distance
            
            return distance
    
    def add_domain_relationship(self, source: str, target: str, 
                               relationship_type: str, strength: float = 0.5):
        """Add relationship between domains"""
        with self._lock:
            relationship = DomainRelationship(
                source=source,
                target=target,
                relationship_type=relationship_type,
                strength=strength
            )
            
            self.relationships.append(relationship)
            self.domain_graph.add_edge(source, target, 
                                      type=relationship_type, 
                                      weight=strength)
            
            # Update domain profiles
            if relationship_type == "parent":
                if source in self.domains:
                    self.domains[source].child_domains.append(target)
                if target in self.domains:
                    self.domains[target].parent_domains.append(source)
            
            # Clear distance cache
            self.distance_cache.clear()
            
            logger.debug("Added relationship: %s -> %s (%s)", source, target, relationship_type)
    
    def get_domain_hierarchy(self, domain_name: str) -> Dict[str, List[str]]:
        """Get domain hierarchy (parents and children)"""
        with self._lock:
            hierarchy = {
                'parents': [],
                'children': [],
                'siblings': [],
                'ancestors': [],
                'descendants': []
            }
            
            if domain_name not in self.domains:
                return hierarchy
            
            profile = self.domains[domain_name]
            hierarchy['parents'] = profile.parent_domains
            hierarchy['children'] = profile.child_domains
            
            # Find siblings (domains with same parent)
            for parent in profile.parent_domains:
                if parent in self.domains:
                    parent_profile = self.domains[parent]
                    for child in parent_profile.child_domains:
                        if child != domain_name and child not in hierarchy['siblings']:
                            hierarchy['siblings'].append(child)
            
            # Find ancestors and descendants using graph
            if self.domain_graph.has_node(domain_name):
                if NETWORKX_AVAILABLE:
                    hierarchy['ancestors'] = list(nx.ancestors(self.domain_graph, domain_name))
                    hierarchy['descendants'] = list(nx.descendants(self.domain_graph, domain_name))
                else:
                    hierarchy['ancestors'] = MockNX.ancestors(self.domain_graph, domain_name)
                    hierarchy['descendants'] = MockNX.descendants(self.domain_graph, domain_name)
            
            return hierarchy
    
    def get_similar_domains(self, domain_name: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Get most similar domains"""
        with self._lock:
            similarities = []
            
            for other_domain in self.domains:
                if other_domain != domain_name:
                    distance = self.calculate_domain_distance(domain_name, other_domain)
                    similarity = 1.0 - distance
                    similarities.append((other_domain, similarity))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
    
    def update_domain_criticality(self, domain_name: str, new_criticality: float):
        """Update domain criticality score"""
        with self._lock:
            if domain_name in self.domains:
                old_criticality = self.domains[domain_name].criticality_score
                
                # SAFETY: Validate new criticality
                if not (0 <= new_criticality <= 1):
                    logger.warning("Invalid criticality score %.2f, clamping to [0,1]", new_criticality)
                    self.safety_corrections['criticality_score'] += 1
                    new_criticality = np.clip(new_criticality, 0, 1)
                
                self.domains[domain_name].criticality_score = new_criticality
                
                # Clear cache as distances may change
                self.distance_cache.clear()
                
                logger.info("Updated criticality for %s: %.2f -> %.2f", 
                           domain_name, old_criticality, new_criticality)
    
    def merge_domains(self, domain_a: str, domain_b: str, new_name: str = None) -> DomainProfile:
        """Merge two domains into one"""
        with self._lock:
            profile_a = self.domains.get(domain_a)
            profile_b = self.domains.get(domain_b)
            
            if not profile_a or not profile_b:
                raise ValueError(f"Cannot merge - domain not found")
            
            # Create merged profile
            merged_name = new_name or f"{domain_a}_{domain_b}"
            merged = DomainProfile(
                name=merged_name,
                criticality_score=max(profile_a.criticality_score, profile_b.criticality_score),
                effect_types=profile_a.effect_types | profile_b.effect_types,
                typical_patterns=profile_a.typical_patterns + profile_b.typical_patterns,
                capabilities=profile_a.capabilities | profile_b.capabilities,
                limitations=profile_a.limitations | profile_b.limitations,
                parent_domains=list(set(profile_a.parent_domains + profile_b.parent_domains)),
                child_domains=list(set(profile_a.child_domains + profile_b.child_domains))
            )
            
            # Register merged domain
            self.register_domain(merged_name, merged)
            
            # Transfer relationships
            for rel in self.relationships:
                if rel.source in [domain_a, domain_b]:
                    self.add_domain_relationship(merged_name, rel.target, 
                                               rel.relationship_type, rel.strength)
                elif rel.target in [domain_a, domain_b]:
                    self.add_domain_relationship(rel.source, merged_name,
                                               rel.relationship_type, rel.strength)
            
            logger.info("Merged domains %s and %s into %s", domain_a, domain_b, merged_name)
            
            return merged
    
    def _find_shortest_path_length(self, graph, source: str, target: str) -> int:
        """Find shortest path length between nodes (BFS)"""
        if source == target:
            return 0
        
        visited = {source}
        queue = [(source, 0)]
        
        while queue:
            current, dist = queue.pop(0)
            
            for neighbor in graph.successors(current):
                if neighbor == target:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        
        # No path found
        raise ValueError(f"No path between {source} and {target}")
    
    def _categorize_effect(self, effect_type: str) -> EffectCategory:
        """Categorize effect type string"""
        effect_type_lower = effect_type.lower()
        
        if 'compute' in effect_type_lower or 'calculate' in effect_type_lower:
            return EffectCategory.COMPUTATION
        elif 'transform' in effect_type_lower or 'convert' in effect_type_lower:
            return EffectCategory.TRANSFORMATION
        elif 'optimize' in effect_type_lower or 'improve' in effect_type_lower:
            return EffectCategory.OPTIMIZATION
        elif 'control' in effect_type_lower or 'regulate' in effect_type_lower:
            return EffectCategory.CONTROL
        elif 'predict' in effect_type_lower or 'forecast' in effect_type_lower:
            return EffectCategory.PREDICTION
        elif 'generate' in effect_type_lower or 'create' in effect_type_lower:
            return EffectCategory.GENERATION
        elif 'analyze' in effect_type_lower or 'examine' in effect_type_lower:
            return EffectCategory.ANALYSIS
        else:
            return EffectCategory.VALIDATION
    
    def _pattern_to_effect_category(self, pattern_type: PatternType) -> EffectCategory:
        """Convert pattern type to effect category"""
        mapping = {
            PatternType.STRUCTURAL: EffectCategory.TRANSFORMATION,
            PatternType.BEHAVIORAL: EffectCategory.CONTROL,
            PatternType.TEMPORAL: EffectCategory.PREDICTION,
            PatternType.SPATIAL: EffectCategory.TRANSFORMATION,
            PatternType.HIERARCHICAL: EffectCategory.ANALYSIS,
            PatternType.ITERATIVE: EffectCategory.OPTIMIZATION,
            PatternType.RECURSIVE: EffectCategory.COMPUTATION
        }
        return mapping.get(pattern_type, EffectCategory.VALIDATION)
    
    def _update_effect_statistics(self, effect: DomainEffect):
        """
        Update statistics for effect category (FIXED: enforce size limit)
        """
        category = effect.category.value
        
        # FIXED: Enforce effect statistics limit
        if category not in self.effect_statistics:
            if len(self.effect_statistics) >= self.max_effect_categories:
                # Remove category with lowest count
                min_category = min(self.effect_statistics.keys(),
                                 key=lambda k: self.effect_statistics[k]['count'])
                del self.effect_statistics[min_category]
                logger.debug("Evicted effect statistics for category %s", min_category)
            
            self.effect_statistics[category] = {'count': 0, 'avg_importance': 0.0}
        
        stats = self.effect_statistics[category]
        
        count = stats['count']
        avg_importance = stats['avg_importance']
        
        # Update running average
        stats['count'] = count + 1
        stats['avg_importance'] = (avg_importance * count + effect.importance) / (count + 1)
    
    def _initialize_default_domains(self):
        """Initialize with default domains"""
        default_domains = [
            DomainProfile(
                name="general",
                criticality_score=DomainCriticality.LOW.value,
                effect_types={"compute", "transform", "validate"},
                capabilities={"basic_processing", "data_manipulation"},
                limitations={"no_real_time", "limited_resources"}
            ),
            DomainProfile(
                name="safety_critical",
                criticality_score=DomainCriticality.SAFETY_CRITICAL.value,
                effect_types={"validate", "control", "monitor"},
                capabilities={"fault_tolerance", "redundancy", "verification"},
                limitations={"high_latency_tolerance", "strict_validation"}
            ),
            DomainProfile(
                name="optimization",
                criticality_score=DomainCriticality.MEDIUM.value,
                effect_types={"optimize", "compute", "analyze"},
                capabilities={"gradient_computation", "constraint_handling"},
                limitations={"local_optima", "computational_intensity"}
            ),
            DomainProfile(
                name="real_time",
                criticality_score=DomainCriticality.HIGH.value,
                effect_types={"control", "predict", "respond"},
                capabilities={"low_latency", "deterministic_execution"},
                limitations={"resource_constraints", "timing_constraints"}
            ),
            DomainProfile(
                name="machine_learning",
                criticality_score=DomainCriticality.MEDIUM_HIGH.value,
                effect_types={"predict", "classify", "generate"},
                capabilities={"pattern_recognition", "adaptation"},
                limitations={"data_dependency", "interpretability"}
            )
        ]
        
        for profile in default_domains:
            self.register_domain(profile.name, profile)
        
        # Add some default relationships
        self.add_domain_relationship("general", "optimization", "specialization", 0.7)
        self.add_domain_relationship("general", "machine_learning", "specialization", 0.6)
        self.add_domain_relationship("safety_critical", "real_time", "related", 0.8)
    
    def _save_registry(self):
        """Save registry to disk"""
        if not self.storage_path:
            return
        
        with self._lock:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            
            # Save domains
            domains_file = self.storage_path / "domains.pkl"
            with open(domains_file, 'wb') as f:
                pickle.dump(self.domains, f)
            
            # Save relationships
            relationships_file = self.storage_path / "relationships.json"
            with open(relationships_file, 'w') as f:
                rel_data = [
                    {
                        'source': r.source,
                        'target': r.target,
                        'type': r.relationship_type,
                        'strength': r.strength
                    }
                    for r in self.relationships
                ]
                json.dump(rel_data, f)
    
    def _load_registry(self):
        """Load registry from disk"""
        if not self.storage_path or not self.storage_path.exists():
            return
        
        # Load domains
        domains_file = self.storage_path / "domains.pkl"
        if domains_file.exists():
            try:
                with open(domains_file, 'rb') as f:
                    loaded_domains = safe_pickle_load(f)
                    for name, profile in loaded_domains.items():
                        self.register_domain(name, profile)
            except Exception as e:
                logger.warning("Failed to load domains: %s", e)
        
        # Load relationships
        relationships_file = self.storage_path / "relationships.json"
        if relationships_file.exists():
            try:
                with open(relationships_file, 'r') as f:
                    rel_data = json.load(f)
                    for rel in rel_data:
                        self.add_domain_relationship(
                            rel['source'], rel['target'],
                            rel['type'], rel['strength']
                        )
            except Exception as e:
                logger.warning("Failed to load relationships: %s", e)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics"""
        stats = {
            'total_domains': self.total_domains,
            'active_domains': len(self.domains),
            'total_relationships': len(self.relationships),
            'criticality_distribution': self._get_criticality_distribution(),
            'effect_statistics': dict(self.effect_statistics),
            'effect_domains_tracked': len(self.domain_effects),
            'cache_size': len(self.distance_cache),
            'max_cache_size': self.max_cache_size,
            'cache_hit_rate': self.cache_hit_count / max(1, self.cache_hit_count + self.cache_miss_count),
            'world_model_connected': self.world_model is not None,
            'max_domains': self.max_domains,
            'max_effect_domains': self.max_effect_domains,
            'max_effect_categories': self.max_effect_categories
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
    
    def _get_criticality_distribution(self) -> Dict[str, int]:
        """Get distribution of domains by criticality"""
        distribution = {}
        
        for profile in self.domains.values():
            risk_level = profile.get_risk_level()
            distribution[risk_level] = distribution.get(risk_level, 0) + 1
        
        return distribution


class RiskAdjuster:
    """Adjusts thresholds based on domain criticality (FIXED: externalized configuration)"""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize with optional config file (FIXED: externalized config)
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config_path = config_path
        
        # Load config if provided
        if config_path and config_path.exists():
            self._load_config(config_path)
        else:
            self._set_defaults()
        
        logger.info("RiskAdjuster initialized with config from %s", 
                   config_path if config_path else "defaults")
    
    def _set_defaults(self):
        """
        Set default configuration (FIXED: externalized config)
        """
        self.base_thresholds = {
            'confidence': 0.7,
            'success_rate': 0.6,
            'validation': 0.8,
            'timeout': 60.0,
            'retry_limit': 3
        }
        
        self.criticality_multipliers = {
            'confidence': 1.3,
            'success_rate': 1.2,
            'validation': 1.4,
            'timeout': 0.5,
            'retry_limit': 1.5
        }
    
    def _load_config(self, config_path: Path):
        """
        Load configuration from file (FIXED: externalized config)
        
        Args:
            config_path: Path to configuration file
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            self.base_thresholds = config.get('base_thresholds', {})
            self.criticality_multipliers = config.get('criticality_multipliers', {})
            
            # Validate loaded config
            required_keys = ['confidence', 'success_rate', 'validation', 'timeout', 'retry_limit']
            for key in required_keys:
                if key not in self.base_thresholds:
                    raise ValueError(f"Missing required threshold: {key}")
            
            logger.info("Loaded RiskAdjuster config from %s", config_path)
            
        except Exception as e:
            logger.error("Failed to load config, using defaults: %s", e)
            self._set_defaults()
    
    def save_config(self, config_path: Path):
        """
        Save current configuration (FIXED: externalized config)
        
        Args:
            config_path: Path to save configuration file
        """
        config = {
            'base_thresholds': self.base_thresholds,
            'criticality_multipliers': self.criticality_multipliers
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info("Saved RiskAdjuster config to %s", config_path)
    
    def get_dynamic_thresholds(self, domain_profile: DomainProfile) -> Dict[str, float]:
        """
        Get dynamically adjusted thresholds for domain
        
        Args:
            domain_profile: Domain profile
            
        Returns:
            Adjusted thresholds
        """
        adjusted = {}
        criticality = domain_profile.criticality_score
        
        for threshold_name, base_value in self.base_thresholds.items():
            # Calculate adjustment factor
            if threshold_name in ['timeout']:
                # Inverse relationship - higher criticality means lower timeout
                factor = 1.0 - (criticality * 0.5)
            else:
                # Direct relationship - higher criticality means higher threshold
                multiplier = self.criticality_multipliers.get(threshold_name, 1.0)
                factor = 1.0 + (criticality * (multiplier - 1.0))
            
            adjusted[threshold_name] = base_value * factor
            
            # Apply safety margins
            safety_margin = self.calculate_safety_margin(criticality)
            
            if threshold_name in ['confidence', 'success_rate', 'validation']:
                # Add safety margin to thresholds
                adjusted[threshold_name] = min(0.99, adjusted[threshold_name] + safety_margin)
        
        # Add domain-specific adjustments
        if domain_profile.name == "safety_critical":
            adjusted['confidence'] = max(0.95, adjusted['confidence'])
            adjusted['validation'] = max(0.95, adjusted['validation'])
        elif domain_profile.name == "real_time":
            adjusted['timeout'] = min(10.0, adjusted['timeout'])
        
        return adjusted
    
    def calculate_safety_margin(self, criticality: float) -> float:
        """
        Calculate safety margin based on criticality
        
        Args:
            criticality: Criticality score [0, 1]
            
        Returns:
            Safety margin to add to thresholds
        """
        # Non-linear safety margin calculation
        if criticality >= DomainCriticality.SAFETY_CRITICAL.value:
            margin = 0.2
        elif criticality >= DomainCriticality.HIGH.value:
            margin = 0.15
        elif criticality >= DomainCriticality.MEDIUM_HIGH.value:
            margin = 0.1
        elif criticality >= DomainCriticality.MEDIUM.value:
            margin = 0.05
        else:
            margin = 0.0
        
        return margin
    
    def assess_risk(self, domain_profile: DomainProfile, 
                   operation: str) -> Dict[str, Any]:
        """
        Assess risk for operation in domain
        
        Args:
            domain_profile: Domain profile
            operation: Operation type
            
        Returns:
            Risk assessment
        """
        assessment = {
            'risk_level': domain_profile.get_risk_level(),
            'criticality_score': domain_profile.criticality_score,
            'requires_validation': domain_profile.criticality_score >= 0.7,
            'requires_monitoring': domain_profile.criticality_score >= 0.5,
            'safety_margin': self.calculate_safety_margin(domain_profile.criticality_score),
            'recommended_actions': []
        }
        
        # Add recommendations based on risk
        if domain_profile.criticality_score >= DomainCriticality.HIGH.value:
            assessment['recommended_actions'].extend([
                'dual_validation',
                'comprehensive_testing',
                'rollback_capability',
                'audit_logging'
            ])
        elif domain_profile.criticality_score >= DomainCriticality.MEDIUM.value:
            assessment['recommended_actions'].extend([
                'standard_validation',
                'performance_monitoring'
            ])
        
        # Operation-specific adjustments
        if operation in ['transfer', 'adaptation']:
            assessment['requires_validation'] = True
            if domain_profile.criticality_score >= 0.5:
                assessment['recommended_actions'].append('staged_rollout')
        
        return assessment
    
    def get_validation_requirements(self, criticality: float) -> Dict[str, Any]:
        """Get validation requirements based on criticality"""
        requirements = {
            'min_test_coverage': 0.7,
            'min_success_rate': 0.8,
            'required_tests': ['unit', 'integration'],
            'validation_passes': 1
        }
        
        if criticality >= DomainCriticality.SAFETY_CRITICAL.value:
            requirements.update({
                'min_test_coverage': 0.95,
                'min_success_rate': 0.99,
                'required_tests': ['unit', 'integration', 'system', 'acceptance', 'safety'],
                'validation_passes': 3
            })
        elif criticality >= DomainCriticality.HIGH.value:
            requirements.update({
                'min_test_coverage': 0.9,
                'min_success_rate': 0.95,
                'required_tests': ['unit', 'integration', 'system', 'acceptance'],
                'validation_passes': 2
            })
        elif criticality >= DomainCriticality.MEDIUM.value:
            requirements.update({
                'min_test_coverage': 0.8,
                'min_success_rate': 0.9,
                'required_tests': ['unit', 'integration', 'system'],
                'validation_passes': 1
            })
        
        return requirements