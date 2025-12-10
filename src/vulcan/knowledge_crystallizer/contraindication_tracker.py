"""
contraindication_tracker.py - Contraindication tracking for Knowledge Crystallizer
Part of the VULCAN-AGI system
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque, Counter
import time
import json
from pathlib import Path
from enum import Enum
import threading
import pickle
import hashlib
from ..security_fixes import safe_pickle_load

# Optional imports with fallbacks
try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logging.warning("networkx not available, graph features will be limited")

logger = logging.getLogger(__name__)


class FailureMode(Enum):
    """Types of failure modes"""

    PERFORMANCE = "performance"
    CORRECTNESS = "correctness"
    STABILITY = "stability"
    RESOURCE = "resource"
    DOMAIN_SPECIFIC = "domain_specific"
    CASCADING = "cascading"
    TIMEOUT = "timeout"
    DEPENDENCY = "dependency"
    COMPATIBILITY = "compatibility"


class Severity(Enum):
    """Severity levels for contraindications"""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    @classmethod
    def from_score(cls, score: float) -> "Severity":
        """Convert numeric score to severity level"""
        if score < 0.25:
            return cls.LOW
        elif score < 0.5:
            return cls.MEDIUM
        elif score < 0.75:
            return cls.HIGH
        else:
            return cls.CRITICAL


@dataclass
class Contraindication:
    """Single contraindication specification"""

    condition: str
    failure_mode: str
    frequency: int = 0
    severity: float = 0.5
    workaround: Optional[str] = None
    domain: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    principle_id: Optional[str] = None  # Associated principle
    confidence: float = 0.8  # Confidence in this contraindication

    def get_severity_level(self) -> Severity:
        """Get severity as enum level"""
        return Severity.from_score(self.severity)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "condition": self.condition,
            "failure_mode": self.failure_mode,
            "frequency": self.frequency,
            "severity": self.severity,
            "workaround": self.workaround,
            "domain": self.domain,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "principle_id": self.principle_id,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Contraindication":
        """Create from dictionary"""
        return cls(**data)

    def update_frequency(self, increment: int = 1):
        """Update frequency count"""
        self.frequency += increment
        # Increase severity slightly with frequency
        self.severity = min(1.0, self.severity * (1 + 0.01 * increment))

    def merge_with(self, other: "Contraindication") -> "Contraindication":
        """Merge with another contraindication"""
        if self.condition != other.condition:
            raise ValueError("Can only merge contraindications with same condition")

        return Contraindication(
            condition=self.condition,
            failure_mode=self.failure_mode,
            frequency=self.frequency + other.frequency,
            severity=max(self.severity, other.severity),
            workaround=self.workaround or other.workaround,
            domain=self.domain or other.domain,
            timestamp=min(self.timestamp, other.timestamp),
            metadata={**self.metadata, **other.metadata},
            principle_id=self.principle_id or other.principle_id,
            confidence=(self.confidence + other.confidence) / 2,
        )


@dataclass
class CascadeImpact:
    """Impact of cascading contraindications"""

    affected_principles: List[Any] = field(default_factory=list)
    impact_scores: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    max_severity: float = 0.0
    total_impact: float = 0.0
    cascade_depth: int = 0
    mitigation_strategies: List[str] = field(default_factory=list)
    recovery_time_estimate: float = 0.0  # in seconds
    blast_radius: int = 0  # Number of affected components

    def add_affected(
        self, principle, impact_score: float, mitigation: Optional[str] = None
    ):
        """Add an affected principle"""
        self.affected_principles.append(principle)
        principle_id = getattr(principle, "id", str(principle))
        self.impact_scores[principle_id] = impact_score
        self.max_severity = max(self.max_severity, impact_score)
        self.total_impact += impact_score
        self.blast_radius += 1

        if mitigation:
            self.warnings.append(f"Principle {principle_id}: {mitigation}")
            if mitigation not in self.mitigation_strategies:
                self.mitigation_strategies.append(mitigation)

    def add_warning(self, warning: str):
        """Add a warning message"""
        self.warnings.append(warning)

    def estimate_recovery_time(self):
        """Estimate recovery time based on impact"""
        base_time = 10.0  # Base recovery time in seconds
        severity_multiplier = 1 + self.max_severity * 2
        depth_multiplier = 1 + self.cascade_depth * 0.5
        blast_multiplier = 1 + self.blast_radius * 0.1

        self.recovery_time_estimate = (
            base_time * severity_multiplier * depth_multiplier * blast_multiplier
        )
        return self.recovery_time_estimate

    def get_risk_level(self) -> str:
        """Get overall risk level"""
        if self.max_severity > 0.8 or self.blast_radius > 20:
            return "CRITICAL"
        elif self.max_severity > 0.6 or self.blast_radius > 10:
            return "HIGH"
        elif self.max_severity > 0.4 or self.blast_radius > 5:
            return "MEDIUM"
        else:
            return "LOW"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "affected_principles": [
                getattr(p, "id", str(p)) for p in self.affected_principles
            ],
            "impact_scores": self.impact_scores,
            "warnings": self.warnings,
            "max_severity": self.max_severity,
            "total_impact": self.total_impact,
            "cascade_depth": self.cascade_depth,
            "mitigation_strategies": self.mitigation_strategies,
            "recovery_time_estimate": self.recovery_time_estimate,
            "blast_radius": self.blast_radius,
            "risk_level": self.get_risk_level(),
        }


class ContraindicationDatabase:
    """Manages contraindications for principles"""

    def __init__(self, persistence_path: Optional[Path] = None):
        """
        Initialize contraindication database

        Args:
            persistence_path: Optional path for persisting database
        """
        self.contraindications = defaultdict(list)  # principle_id -> [Contraindication]
        self.domain_contraindications = defaultdict(
            set
        )  # domain -> set of principle_ids
        self.failure_patterns = defaultdict(list)  # failure_mode -> [principle_id, ...]
        self.condition_index = defaultdict(set)  # condition -> set of principle_ids

        # Statistics
        self.total_contraindications = 0
        self.contraindication_history = deque(maxlen=10000)
        self.failure_counts = Counter()  # Track failure mode frequencies

        # Thread safety - MUST BE INITIALIZED BEFORE LOADING
        self.lock = threading.RLock()

        # Persistence
        self.persistence_path = persistence_path
        if persistence_path and persistence_path.exists():
            self.load()

        logger.info("ContraindicationDatabase initialized")

    def register(self, principle_id: str, contraindication: Contraindication):
        """
        Register a contraindication for a principle

        Args:
            principle_id: ID of the principle
            contraindication: Contraindication to register
        """
        with self.lock:
            # Set principle ID in contraindication
            contraindication.principle_id = principle_id

            # Check for existing similar contraindication
            existing = self._find_similar_contraindication(
                principle_id, contraindication
            )

            if existing:
                # Update existing contraindication
                existing.update_frequency()
                logger.debug(
                    "Updated existing contraindication for principle %s", principle_id
                )
            else:
                # Add new contraindication
                self.contraindications[principle_id].append(contraindication)

                # Index by domain
                if contraindication.domain:
                    self.domain_contraindications[contraindication.domain].add(
                        principle_id
                    )

                # Index by failure mode
                self.failure_patterns[contraindication.failure_mode].append(
                    principle_id
                )

                # Index by condition
                self.condition_index[contraindication.condition].add(principle_id)

                # Update statistics
                self.total_contraindications += 1
                self.failure_counts[contraindication.failure_mode] += 1

            # Add to history
            self.contraindication_history.append(
                {
                    "principle_id": principle_id,
                    "contraindication": contraindication.to_dict(),
                    "timestamp": time.time(),
                }
            )

            logger.debug(
                "Registered contraindication for principle %s: %s",
                principle_id,
                contraindication.condition,
            )

            # Auto-save if persistence enabled
            if self.persistence_path:
                self.save()

    def batch_register(self, contraindications: List[Tuple[str, Contraindication]]):
        """
        Register multiple contraindications efficiently

        Args:
            contraindications: List of (principle_id, contraindication) tuples
        """
        with self.lock:
            for principle_id, contraindication in contraindications:
                self.register(principle_id, contraindication)

    def get_contraindications(self, principle_id: str) -> List[Contraindication]:
        """
        Get all contraindications for a principle

        Args:
            principle_id: ID of the principle

        Returns:
            List of contraindications
        """
        with self.lock:
            return self.contraindications.get(principle_id, []).copy()

    def get_by_condition(self, condition: str) -> Dict[str, List[Contraindication]]:
        """
        Get all contraindications with specific condition

        Args:
            condition: Condition to search for

        Returns:
            Dictionary mapping principle IDs to contraindications
        """
        with self.lock:
            result = {}
            for principle_id in self.condition_index.get(condition, set()):
                contras = [
                    c
                    for c in self.contraindications[principle_id]
                    if c.condition == condition
                ]
                if contras:
                    result[principle_id] = contras
            return result

    def check_domain_compatibility(
        self, principle_id: str, domain: str
    ) -> Tuple[bool, List[Contraindication]]:
        """
        Check if principle is compatible with domain

        Args:
            principle_id: ID of the principle
            domain: Domain to check

        Returns:
            Tuple of (is_compatible, blocking_contraindications)
        """
        with self.lock:
            contraindications = self.get_contraindications(principle_id)

            blocking = []
            for contraindication in contraindications:
                if contraindication.domain == domain:
                    # Direct domain contraindication
                    blocking.append(contraindication)
                elif contraindication.domain and self._domains_related(
                    contraindication.domain, domain
                ):
                    # Related domain contraindication
                    blocking.append(contraindication)

            # Check severity
            critical_blocking = [
                c for c in blocking if c.get_severity_level() == Severity.CRITICAL
            ]

            # Not compatible if there are critical contraindications
            is_compatible = len(critical_blocking) == 0

            return is_compatible, blocking

    def get_domain_contraindicated_principles(self, domain: str) -> Set[str]:
        """
        Get all principles contraindicated for a domain

        Args:
            domain: Domain name

        Returns:
            Set of principle IDs
        """
        with self.lock:
            direct = self.domain_contraindications.get(domain, set()).copy()

            # Also check related domains
            related = set()
            for check_domain, principles in self.domain_contraindications.items():
                if self._domains_related(check_domain, domain):
                    related.update(principles)

            return direct | related

    def get_failure_pattern_principles(self, failure_mode: str) -> List[str]:
        """
        Get principles with specific failure mode

        Args:
            failure_mode: Type of failure

        Returns:
            List of principle IDs
        """
        with self.lock:
            return self.failure_patterns.get(failure_mode, []).copy()

    def analyze_failure_patterns(self) -> Dict[str, Any]:
        """
        Analyze patterns in failures

        Returns:
            Analysis results
        """
        with self.lock:
            # Find common failure combinations
            failure_combinations = defaultdict(int)
            for principle_id, contras in self.contraindications.items():
                if len(contras) > 1:
                    modes = tuple(sorted(c.failure_mode for c in contras))
                    failure_combinations[modes] += 1

            # Find high-risk domains
            domain_risks = {}
            for domain, principles in self.domain_contraindications.items():
                risk_score = 0
                for pid in principles:
                    contras = self.contraindications[pid]
                    risk_score += sum(c.severity for c in contras)
                domain_risks[domain] = risk_score / max(1, len(principles))

            return {
                "most_common_failures": self.failure_counts.most_common(10),
                "failure_combinations": dict(failure_combinations),
                "high_risk_domains": sorted(
                    domain_risks.items(), key=lambda x: x[1], reverse=True
                )[:5],
                "total_patterns": len(self.failure_patterns),
            }

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get contraindication statistics

        Returns:
            Dictionary of statistics
        """
        with self.lock:
            severity_counts = defaultdict(int)
            total_frequency = 0

            for principle_contras in self.contraindications.values():
                for contra in principle_contras:
                    severity_counts[contra.get_severity_level().name] += 1
                    total_frequency += contra.frequency

            pattern_analysis = self.analyze_failure_patterns()

            return {
                "total_contraindications": self.total_contraindications,
                "principles_with_contraindications": len(self.contraindications),
                "domains_affected": len(self.domain_contraindications),
                "failure_modes": len(self.failure_patterns),
                "severity_distribution": dict(severity_counts),
                "total_frequency": total_frequency,
                "avg_frequency": total_frequency / max(1, self.total_contraindications),
                "recent_registrations": len(self.contraindication_history),
                "pattern_analysis": pattern_analysis,
            }

    def prune_old_contraindications(self, age_days: int = 90):
        """
        Remove old contraindications with low frequency

        Args:
            age_days: Age threshold in days
        """
        with self.lock:
            current_time = time.time()
            age_threshold = age_days * 86400  # Convert to seconds
            pruned_count = 0

            for principle_id in list(self.contraindications.keys()):
                contras = self.contraindications[principle_id]
                kept = []

                for contra in contras:
                    age = current_time - contra.timestamp
                    # Keep if recent or high frequency
                    if age < age_threshold or contra.frequency > 5:
                        kept.append(contra)
                    else:
                        pruned_count += 1
                        self.total_contraindications -= 1

                if kept:
                    self.contraindications[principle_id] = kept
                else:
                    del self.contraindications[principle_id]

            logger.info("Pruned %d old contraindications", pruned_count)
            return pruned_count

    def save(self, path: Optional[Path] = None):
        """Save database to file"""
        save_path = path or self.persistence_path
        if not save_path:
            return

        with self.lock:
            data = {
                "contraindications": {
                    k: [c.to_dict() for c in v]
                    for k, v in self.contraindications.items()
                },
                "total_contraindications": self.total_contraindications,
                "failure_counts": dict(self.failure_counts),
                "timestamp": time.time(),
            }

            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug("Saved contraindication database to %s", save_path)

    def load(self, path: Optional[Path] = None):
        """Load database from file"""
        load_path = path or self.persistence_path
        if not load_path or not load_path.exists():
            return

        with self.lock:
            try:
                with open(load_path, "r") as f:
                    data = json.load(f)

                # Reconstruct contraindications
                self.contraindications.clear()
                for principle_id, contras_data in data["contraindications"].items():
                    self.contraindications[principle_id] = [
                        Contraindication.from_dict(c) for c in contras_data
                    ]

                # Rebuild indices
                self._rebuild_indices()

                self.total_contraindications = data["total_contraindications"]
                self.failure_counts = Counter(data.get("failure_counts", {}))

                logger.info("Loaded contraindication database from %s", load_path)

            except Exception as e:
                logger.error("Failed to load contraindication database: %s", e)

    def _rebuild_indices(self):
        """Rebuild all indices from contraindications"""
        self.domain_contraindications.clear()
        self.failure_patterns.clear()
        self.condition_index.clear()

        for principle_id, contras in self.contraindications.items():
            for contra in contras:
                if contra.domain:
                    self.domain_contraindications[contra.domain].add(principle_id)
                self.failure_patterns[contra.failure_mode].append(principle_id)
                self.condition_index[contra.condition].add(principle_id)

    def _find_similar_contraindication(
        self, principle_id: str, contraindication: Contraindication
    ) -> Optional[Contraindication]:
        """Find similar existing contraindication"""
        for existing in self.contraindications.get(principle_id, []):
            if (
                existing.condition == contraindication.condition
                and existing.failure_mode == contraindication.failure_mode
                and existing.domain == contraindication.domain
            ):
                return existing
        return None

    def _domains_related(self, domain1: str, domain2: str) -> bool:
        """Check if two domains are related"""
        # Simple heuristic - check for common prefixes or suffixes
        if domain1 == domain2:
            return True

        # Check for hierarchical relationship
        if domain1.startswith(domain2 + "_") or domain2.startswith(domain1 + "_"):
            return True

        # Check for common base
        parts1 = domain1.split("_")
        parts2 = domain2.split("_")

        # Common parts indicate relationship
        common = set(parts1) & set(parts2)
        return len(common) > 0


# Simple graph implementation for when networkx is not available
class SimpleGraph:
    """Simple directed graph implementation as fallback"""

    def __init__(self):
        self.nodes = {}
        self.edges = defaultdict(set)
        self.reverse_edges = defaultdict(set)
        self.edge_weights = {}  # (source, target) -> weight

    def add_node(self, node_id, **attrs):
        self.nodes[node_id] = attrs

    def add_edge(self, source, target, weight=1.0, **attrs):
        self.edges[source].add(target)
        self.reverse_edges[target].add(source)
        self.edge_weights[(source, target)] = weight

    def remove_node(self, node_id):
        """Remove node and all its edges"""
        if node_id in self.nodes:
            del self.nodes[node_id]

            # Remove outgoing edges
            if node_id in self.edges:
                for target in self.edges[node_id]:
                    self.reverse_edges[target].discard(node_id)
                    if (node_id, target) in self.edge_weights:
                        del self.edge_weights[(node_id, target)]
                del self.edges[node_id]

            # Remove incoming edges
            if node_id in self.reverse_edges:
                for source in self.reverse_edges[node_id]:
                    self.edges[source].discard(node_id)
                    if (source, node_id) in self.edge_weights:
                        del self.edge_weights[(source, node_id)]
                del self.reverse_edges[node_id]

    def successors(self, node):
        return list(self.edges.get(node, set()))

    def predecessors(self, node):
        return list(self.reverse_edges.get(node, set()))

    def number_of_nodes(self):
        return len(self.nodes)

    def number_of_edges(self):
        return sum(len(targets) for targets in self.edges.values())

    def __contains__(self, node):
        return node in self.nodes

    def descendants(self, node):
        """Get all descendants of a node"""
        visited = set()
        queue = [node]

        while queue:
            current = queue.pop(0)
            for successor in self.successors(current):
                if successor not in visited:
                    visited.add(successor)
                    queue.append(successor)

        return visited

    def ancestors(self, node):
        """Get all ancestors of a node"""
        visited = set()
        queue = [node]

        while queue:
            current = queue.pop(0)
            for predecessor in self.predecessors(current):
                if predecessor not in visited:
                    visited.add(predecessor)
                    queue.append(predecessor)

        return visited

    def shortest_path(self, source, target):
        """Find shortest path between nodes using BFS"""
        if source not in self.nodes or target not in self.nodes:
            raise ValueError("Node not in graph")

        if source == target:
            return [source]

        # BFS for shortest path
        queue = [(source, [source])]
        visited = {source}

        while queue:
            current, path = queue.pop(0)

            for successor in self.successors(current):
                if successor == target:
                    return path + [successor]

                if successor not in visited:
                    visited.add(successor)
                    queue.append((successor, path + [successor]))

        # No path found
        raise ValueError("No path between nodes")

    def get_edge_weight(self, source, target):
        """Get weight of edge"""
        return self.edge_weights.get((source, target), 1.0)


class ContraindicationGraph:
    """Tracks cascading contraindication effects"""

    def __init__(self, persistence_path: Optional[Path] = None):
        """
        Initialize contraindication graph

        Args:
            persistence_path: Optional path for persistence
        """
        if NETWORKX_AVAILABLE:
            self.graph = nx.DiGraph()
        else:
            self.graph = SimpleGraph()

        self.principle_nodes = {}  # principle_id -> principle object
        self.impact_weights = {}  # (source, target) -> impact score

        # Cascade tracking
        self.cascade_cache = {}
        self.cache_size = 100

        # Thread safety - MUST BE INITIALIZED BEFORE LOADING
        self.lock = threading.RLock()

        # Persistence
        self.persistence_path = persistence_path
        if persistence_path and persistence_path.exists():
            self.load()

        logger.info(
            "ContraindicationGraph initialized (networkx: %s)", NETWORKX_AVAILABLE
        )

    def add_node(self, principle):
        """
        Add principle node to graph

        Args:
            principle: Principle object
        """
        with self.lock:
            principle_id = getattr(principle, "id", str(principle))
            self.principle_nodes[principle_id] = principle
            self.graph.add_node(principle_id, data=principle)

            logger.debug("Added principle %s to contraindication graph", principle_id)

    def add_edge(self, source_id: str, target_id: str, impact: float):
        """
        Add dependency edge with impact score

        Args:
            source_id: Source principle ID
            target_id: Target principle ID
            impact: Impact score [0, 1]
        """
        with self.lock:
            self.graph.add_edge(source_id, target_id, weight=impact)
            self.impact_weights[(source_id, target_id)] = impact

            # Clear cache when graph changes
            self.cascade_cache.clear()

            logger.debug(
                "Added edge %s -> %s (impact=%.2f)", source_id, target_id, impact
            )

    def remove_node(self, principle_id: str):
        """
        Remove principle node from graph

        Args:
            principle_id: Principle ID to remove
        """
        with self.lock:
            if principle_id in self.principle_nodes:
                del self.principle_nodes[principle_id]

            if NETWORKX_AVAILABLE and principle_id in self.graph:
                self.graph.remove_node(principle_id)
            elif not NETWORKX_AVAILABLE:
                self.graph.remove_node(principle_id)

            # Clear cache
            self.cascade_cache.clear()

            logger.debug(
                "Removed principle %s from contraindication graph", principle_id
            )

    def find_cascades(self, principle_id: str, max_depth: int = 3) -> List[List[str]]:
        """
        Find cascade paths from principle

        Args:
            principle_id: Starting principle ID
            max_depth: Maximum cascade depth

        Returns:
            List of cascade paths
        """
        with self.lock:
            # Check cache
            cache_key = (principle_id, max_depth)
            if cache_key in self.cascade_cache:
                return self.cascade_cache[cache_key]

            cascades = []

            if principle_id not in self.graph:
                return cascades

            # BFS to find all paths
            queue = deque([(principle_id, [principle_id])])
            visited_paths = set()

            while queue:
                current, path = queue.popleft()

                if len(path) > max_depth:
                    continue

                # Check for successors
                for successor in self.graph.successors(current):
                    new_path = path + [successor]
                    path_key = tuple(new_path)

                    if path_key not in visited_paths:
                        visited_paths.add(path_key)
                        cascades.append(new_path)
                        queue.append((successor, new_path))

            # Cache result
            if len(self.cascade_cache) < self.cache_size:
                self.cascade_cache[cache_key] = cascades

            return cascades

    def calculate_cascade_risk(self, principle_id: str) -> float:
        """
        Calculate overall cascade risk for principle

        Args:
            principle_id: Principle ID

        Returns:
            Cascade risk score [0, 1]
        """
        with self.lock:
            if principle_id not in self.graph:
                return 0.0

            # Find all cascades
            cascades = self.find_cascades(principle_id)

            if not cascades:
                return 0.0

            # Calculate risk based on cascade properties
            total_risk = 0.0

            for cascade in cascades:
                # Calculate path risk
                path_risk = 1.0

                for i in range(len(cascade) - 1):
                    edge_impact = self.impact_weights.get(
                        (cascade[i], cascade[i + 1]), 0.5
                    )
                    path_risk *= edge_impact

                # Attenuate by depth
                depth_factor = 0.8 ** (len(cascade) - 1)
                path_risk *= depth_factor

                total_risk += path_risk

            # Normalize
            normalized_risk = min(1.0, total_risk / max(1, len(cascades)))

            return normalized_risk

    def get_downstream_principles(self, principle_id: str) -> Set[str]:
        """
        Get all downstream principles

        Args:
            principle_id: Principle ID

        Returns:
            Set of downstream principle IDs
        """
        with self.lock:
            if principle_id not in self.graph:
                return set()

            # Get all descendants
            try:
                if NETWORKX_AVAILABLE:
                    descendants = nx.descendants(self.graph, principle_id)
                else:
                    descendants = self.graph.descendants(principle_id)
                return set(descendants)
            except Exception as e:
                return set()

    def get_upstream_principles(self, principle_id: str) -> Set[str]:
        """
        Get all upstream principles

        Args:
            principle_id: Principle ID

        Returns:
            Set of upstream principle IDs
        """
        with self.lock:
            if principle_id not in self.graph:
                return set()

            # Get all ancestors
            try:
                if NETWORKX_AVAILABLE:
                    ancestors = nx.ancestors(self.graph, principle_id)
                else:
                    ancestors = self.graph.ancestors(principle_id)
                return set(ancestors)
            except Exception as e:
                return set()

    def get_impact_path(
        self, source_id: str, target_id: str
    ) -> Tuple[List[str], float]:
        """
        Get highest impact path between principles

        Args:
            source_id: Source principle ID
            target_id: Target principle ID

        Returns:
            Tuple of (path, total_impact)
        """
        with self.lock:
            if source_id not in self.graph or target_id not in self.graph:
                return [], 0.0

            try:
                # Find shortest path
                if NETWORKX_AVAILABLE:
                    path = nx.shortest_path(self.graph, source_id, target_id)
                else:
                    path = self.graph.shortest_path(source_id, target_id)

                # Calculate total impact
                total_impact = 1.0
                for i in range(len(path) - 1):
                    impact = self.impact_weights.get((path[i], path[i + 1]), 0.5)
                    total_impact *= impact

                return path, total_impact
            except Exception as e:
                return [], 0.0

    def find_critical_nodes(self, threshold: float = 0.7) -> List[str]:
        """
        Find critical nodes with high cascade risk

        Args:
            threshold: Risk threshold

        Returns:
            List of critical principle IDs
        """
        with self.lock:
            critical = []

            for principle_id in self.principle_nodes:
                risk = self.calculate_cascade_risk(principle_id)
                if risk > threshold:
                    critical.append(principle_id)

            return sorted(
                critical, key=lambda p: self.calculate_cascade_risk(p), reverse=True
            )

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        with self.lock:
            if NETWORKX_AVAILABLE:
                num_nodes = self.graph.number_of_nodes()
                num_edges = self.graph.number_of_edges()
            else:
                num_nodes = self.graph.number_of_nodes()
                num_edges = self.graph.number_of_edges()

            critical_nodes = self.find_critical_nodes()

            return {
                "total_nodes": num_nodes,
                "total_edges": num_edges,
                "critical_nodes": len(critical_nodes),
                "cache_size": len(self.cascade_cache),
                "avg_impact": np.mean(list(self.impact_weights.values()))
                if self.impact_weights
                else 0,
            }

    def save(self, path: Optional[Path] = None):
        """Save graph to file"""
        save_path = path or self.persistence_path
        if not save_path:
            return

        with self.lock:
            data = {
                "nodes": list(self.principle_nodes.keys()),
                "edges": [(s, t, w) for (s, t), w in self.impact_weights.items()],
                "timestamp": time.time(),
            }

            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path.with_suffix(".json"), "w") as f:
                json.dump(data, f, indent=2)

            # Save principle objects separately - only picklable ones
            picklable_nodes = {}
            for key, value in self.principle_nodes.items():
                try:
                    # Test if picklable
                    pickle.dumps(value)
                    picklable_nodes[key] = value
                except (pickle.PicklingError, TypeError, AttributeError):
                    # Skip unpicklable objects (like Mock)
                    logger.warning(f"Skipping unpicklable principle node: {key}")
                    continue

            if picklable_nodes:
                with open(save_path.with_suffix(".pkl"), "wb") as f:
                    pickle.dump(picklable_nodes, f)

            logger.debug("Saved contraindication graph to %s", save_path)

    def load(self, path: Optional[Path] = None):
        """Load graph from file"""
        load_path = path or self.persistence_path
        if not load_path:
            return

        with self.lock:
            try:
                # Load structure
                with open(load_path.with_suffix(".json"), "r") as f:
                    data = json.load(f)

                # Load principle objects
                if load_path.with_suffix(".pkl").exists():
                    with open(load_path.with_suffix(".pkl"), "rb") as f:
                        self.principle_nodes = safe_pickle_load(f)

                # Rebuild graph
                for node_id in data["nodes"]:
                    self.graph.add_node(node_id)

                for source, target, weight in data["edges"]:
                    self.add_edge(source, target, weight)

                logger.info("Loaded contraindication graph from %s", load_path)

            except Exception as e:
                logger.error("Failed to load contraindication graph: %s", e)


class CascadeAnalyzer:
    """Analyzes multi-hop cascade effects"""

    def __init__(
        self,
        contraindication_db: Optional[ContraindicationDatabase] = None,
        contraindication_graph: Optional[ContraindicationGraph] = None,
    ):
        """
        Initialize cascade analyzer

        Args:
            contraindication_db: Contraindication database
            contraindication_graph: Contraindication graph
        """
        self.db = contraindication_db or ContraindicationDatabase()
        self.graph = contraindication_graph or ContraindicationGraph()

        # Analysis parameters
        self.attenuation_factor = 0.7
        self.min_impact_threshold = 0.1
        self.max_simulation_depth = 5

        # Simulation cache
        self.simulation_cache = {}
        self.cache_ttl = 300  # 5 minutes

        # Thread safety
        self.lock = threading.RLock()

        logger.info("CascadeAnalyzer initialized")

    def analyze_cascade_impact(self, candidate, max_depth: int = 3) -> CascadeImpact:
        """
        Analyze cascade impact of principle failure

        Args:
            candidate: Candidate principle
            max_depth: Maximum cascade depth

        Returns:
            Cascade impact analysis
        """
        with self.lock:
            impact = CascadeImpact()
            impact.cascade_depth = max_depth

            # Get principle ID
            principle_id = getattr(candidate, "id", str(candidate))

            # Find dependent principles
            dependents = self.find_dependent_principles(candidate)

            # Analyze impact on each dependent
            visited = set()
            queue = deque([(dep, 1, 1.0) for dep in dependents])

            while queue:
                current_principle, depth, parent_impact = queue.popleft()

                if depth > max_depth:
                    continue

                current_id = getattr(current_principle, "id", str(current_principle))
                if current_id in visited:
                    continue
                visited.add(current_id)

                # Simulate failure
                failure_scenario = self.simulate_failure(
                    candidate, {"principle": current_principle}
                )

                if self._affects_principle(failure_scenario, current_principle):
                    # Calculate attenuated impact
                    current_impact = (
                        failure_scenario.get("severity", 0.5) * parent_impact
                    )
                    current_impact = self.calculate_attenuation(depth, current_impact)

                    if current_impact > self.min_impact_threshold:
                        # Add to impact
                        mitigation = self.suggest_mitigation(failure_scenario)
                        impact.add_affected(
                            current_principle, current_impact, mitigation
                        )

                        # Find next level dependents
                        next_deps = self.find_dependent_principles(current_principle)
                        for next_dep in next_deps:
                            queue.append((next_dep, depth + 1, current_impact))

            # Estimate recovery time
            impact.estimate_recovery_time()

            # Add warnings for high-risk cascades
            risk_level = impact.get_risk_level()
            if risk_level == "CRITICAL":
                impact.add_warning(
                    "CRITICAL cascade risk - immediate intervention required"
                )
            elif risk_level == "HIGH":
                impact.add_warning("High cascade risk - implement circuit breakers")

            if len(impact.affected_principles) > 10:
                impact.add_warning(
                    f"Wide cascade impact - {len(impact.affected_principles)} principles affected"
                )

            return impact

    def find_dependent_principles(self, candidate) -> List[Any]:
        """
        Find principles that depend on candidate

        Args:
            candidate: Candidate principle

        Returns:
            List of dependent principles
        """
        principle_id = getattr(candidate, "id", str(candidate))

        # Get downstream principles from graph
        downstream_ids = self.graph.get_downstream_principles(principle_id)

        # Get principle objects
        dependents = []
        for dep_id in downstream_ids:
            if dep_id in self.graph.principle_nodes:
                dependents.append(self.graph.principle_nodes[dep_id])

        return dependents

    def simulate_failure(self, candidate, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate principle failure in context

        Args:
            candidate: Failing principle
            context: Simulation context

        Returns:
            Failure scenario details
        """
        # Check cache
        cache_key = (id(candidate), frozenset(context.items()) if context else None)
        if cache_key in self.simulation_cache:
            cached_time, cached_result = self.simulation_cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                return cached_result

        scenario = {
            "principle_id": getattr(candidate, "id", str(candidate)),
            "context": context,
            "failure_type": "cascade",
            "severity": 0.5,
            "affected_components": [],
            "recovery_possible": True,
            "estimated_recovery_time": 10.0,
        }

        # Get contraindications
        contraindications = self.db.get_contraindications(scenario["principle_id"])

        if contraindications:
            # Use worst-case contraindication
            max_severity = max(c.severity for c in contraindications)
            scenario["severity"] = max_severity

            # Check for critical failures
            critical = [
                c
                for c in contraindications
                if c.get_severity_level() == Severity.CRITICAL
            ]
            if critical:
                scenario["failure_type"] = "critical"
                scenario["recovery_possible"] = False
                scenario["estimated_recovery_time"] *= 10

        # Analyze context-specific impacts
        if "principle" in context:
            target = context["principle"]
            target_id = getattr(target, "id", str(target))

            # Check for direct impact path
            path, impact = self.graph.get_impact_path(
                scenario["principle_id"], target_id
            )
            if path:
                scenario["impact_path"] = path
                scenario["path_impact"] = impact
                scenario["severity"] *= impact
                scenario["affected_components"] = path

        # Cache result
        self.simulation_cache[cache_key] = (time.time(), scenario)

        # Clean old cache entries periodically with size enforcement
        if len(self.simulation_cache) > 200:
            self._clean_cache()
            # If still over limit after cleaning, enforce hard limit
            if len(self.simulation_cache) > 200:
                sorted_items = sorted(
                    self.simulation_cache.items(), key=lambda x: x[1][0], reverse=True
                )
                self.simulation_cache = dict(sorted_items[:200])

        return scenario

    def suggest_mitigation(self, failure_scenario: Dict[str, Any]) -> Optional[str]:
        """
        Suggest mitigation for failure scenario

        Args:
            failure_scenario: Failure scenario details

        Returns:
            Mitigation suggestion or None
        """
        suggestions = []

        # Check severity
        severity = failure_scenario.get("severity", 0)

        if severity > 0.8:
            suggestions.append("Implement immediate circuit breaker")
        elif severity > 0.6:
            suggestions.append("Add validation checks before application")
        elif severity > 0.4:
            suggestions.append("Monitor closely during application")

        # Check failure type
        failure_type = failure_scenario.get("failure_type", "")

        if failure_type == "critical":
            suggestions.append("Avoid using in this context")
        elif failure_type == "cascade":
            suggestions.append("Implement cascade prevention checks")

        # Check recovery
        if not failure_scenario.get("recovery_possible", True):
            suggestions.append("Prepare rollback mechanism")

        # Check recovery time
        recovery_time = failure_scenario.get("estimated_recovery_time", 0)
        if recovery_time > 60:
            suggestions.append(f"Plan for {recovery_time:.0f}s recovery time")

        return "; ".join(suggestions) if suggestions else None

    def calculate_attenuation(self, depth: int, base_impact: float) -> float:
        """
        Calculate attenuated impact based on depth

        Args:
            depth: Cascade depth
            base_impact: Base impact score

        Returns:
            Attenuated impact
        """
        # Exponential decay with depth
        attenuated = base_impact * (self.attenuation_factor**depth)

        # Apply minimum threshold
        if attenuated < self.min_impact_threshold:
            attenuated = 0.0

        return attenuated

    def predict_cascade_path(self, principle_id: str, failure_mode: str) -> List[str]:
        """
        Predict most likely cascade path for failure

        Args:
            principle_id: Starting principle
            failure_mode: Type of failure

        Returns:
            Predicted cascade path
        """
        # Get principles with same failure mode
        similar_failures = self.db.get_failure_pattern_principles(failure_mode)

        # Find paths to similar failure principles
        paths = []
        for target_id in similar_failures:
            if target_id != principle_id:
                path, impact = self.graph.get_impact_path(principle_id, target_id)
                if path and impact > 0.3:
                    paths.append((path, impact))

        # Return highest impact path
        if paths:
            paths.sort(key=lambda x: x[1], reverse=True)
            return paths[0][0]

        return [principle_id]  # No cascade predicted

    def _affects_principle(self, failure_scenario: Dict[str, Any], principle) -> bool:
        """Check if failure scenario affects principle"""
        # Simple heuristic - check if there's an impact path
        if "impact_path" in failure_scenario:
            principle_id = getattr(principle, "id", str(principle))
            return principle_id in failure_scenario["impact_path"]

        # Check severity threshold
        return failure_scenario.get("severity", 0) > 0.3

    def _clean_cache(self):
        """Clean old cache entries"""
        current_time = time.time()
        old_keys = []

        for key, (cached_time, _) in self.simulation_cache.items():
            if current_time - cached_time > self.cache_ttl:
                old_keys.append(key)

        for key in old_keys:
            del self.simulation_cache[key]

    def get_cascade_statistics(self) -> Dict[str, Any]:
        """
        Get cascade analysis statistics

        Returns:
            Dictionary of statistics
        """
        with self.lock:
            all_risks = []
            for principle_id in self.graph.principle_nodes:
                risk = self.graph.calculate_cascade_risk(principle_id)
                all_risks.append(risk)

            critical_nodes = self.graph.find_critical_nodes()

            return {
                "total_principles": len(self.graph.principle_nodes),
                "total_edges": self.graph.graph.number_of_edges()
                if hasattr(self.graph.graph, "number_of_edges")
                else 0,
                "avg_cascade_risk": np.mean(all_risks) if all_risks else 0,
                "max_cascade_risk": np.max(all_risks) if all_risks else 0,
                "high_risk_principles": sum(1 for r in all_risks if r > 0.7),
                "critical_nodes": len(critical_nodes),
                "cache_size": len(self.simulation_cache),
                "attenuation_factor": self.attenuation_factor,
            }
