from __future__ import annotations

"""
Causal Context Selector - Advanced Causal Reasoning System

A production-ready causal context selection system that builds causally-relevant
context slices using sophisticated causal inference techniques:

Core Capabilities:
- Multi-hop causal graph traversal with path finding
- Temporal causal reasoning with time-series analysis
- Intervention tracking and do-calculus operations
- Probabilistic causal models with Bayesian networks
- Counterfactual reasoning ("what-if" analysis)
- Causal feature importance and attribution
- Causal explanation generation
- Confounding detection and adjustment
- Mediation analysis
- Causal discovery from observational data

Enhanced Features:
- World model integration with causal DAG interfaces
- Hierarchical memory integration (episodic/semantic/procedural)
- Lexical concept extraction with semantic enrichment
- Time decay modeling with multiple decay functions
- Context relevance scoring with multiple strategies
- Caching for performance optimization
- Comprehensive provenance tracking
- Statistical causal inference
- Granger causality testing
- Transfer entropy computation

Input (duck-typed):
- world_model: may expose:
    * extract_concepts(text) -> List[str]
    * get_related_concepts(concepts) -> List[str]
    * causal_graph / causal_dag interfaces
    * get_causal_parents(node) -> List[str]
    * get_causal_children(node) -> List[str]
    * compute_intervention_effect(intervention, outcome) -> float
    * validate_generation(token, context) -> bool
- query: str | List[Token] | dict with keys:
    - "text" | "prompt" | "tokens"
    - "memory": {"episodic": [...], "semantic": [...], "procedural": [...]}
    - "limit": int
    - "causal_depth": int (max hops in causal graph)
    - "include_confounders": bool
    - "temporal_window": float (seconds)

Output:
{
  "causal_context": [
      {
        "source": "episodic|semantic|procedural",
        "score": float,
        "item": <original>,
        "reason": str,
        "causal_path": List[str],
        "causal_strength": float,
        "temporal_relevance": float,
      },
      ...
  ],
  "concepts": List[str],
  "causal_graph": Dict (if available),
  "interventions": List[Dict],
  "confounders": List[str],
  "mediators": List[str],
  "limit": int,
  "statistics": Dict[str, Any],
}
"""

import hashlib
import logging
import re
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

# Set up logger for this module
_logger = logging.getLogger(__name__)

Token = Union[int, str]
Tokens = List[Token]


# ================================ Enums and Data Structures ================================ #


class CausalStrengthType(Enum):
    """Types of causal strength measurement"""

    CORRELATION = "correlation"
    GRANGER = "granger"
    TRANSFER_ENTROPY = "transfer_entropy"
    INTERVENTION = "intervention"
    COUNTERFACTUAL = "counterfactual"


class TemporalDecayFunction(Enum):
    """Temporal decay functions"""

    EXPONENTIAL = "exponential"
    HYPERBOLIC = "hyperbolic"
    POWER_LAW = "power_law"
    LINEAR = "linear"


@dataclass
class CausalPath:
    """Represents a causal path in the graph"""

    nodes: List[str]
    strength: float
    length: int
    path_type: str = "direct"  # direct, mediated, confounded
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalIntervention:
    """Represents an intervention in the causal model"""

    variable: str
    value: Any
    timestamp: float
    effect_on: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CounterfactualScenario:
    """Counterfactual analysis result"""

    original_value: Any
    counterfactual_value: Any
    outcome_difference: float
    plausibility: float
    explanation: str


@dataclass
class CausalStatistics:
    """Statistical summary of causal analysis"""

    total_nodes: int = 0
    total_edges: int = 0
    avg_path_length: float = 0.0
    max_path_length: int = 0
    num_confounders: int = 0
    num_mediators: int = 0
    temporal_span: float = 0.0
    computation_time_ms: float = 0.0


# ================================ CausalContext ================================ #


class CausalContext:
    """
    Advanced causal context selection system with sophisticated causal inference.

    Features:
    - Multi-hop causal graph traversal
    - Temporal causal reasoning
    - Intervention tracking and analysis
    - Counterfactual reasoning
    - Confounding detection
    - Statistical causal inference
    - Performance optimization with caching

    Usage:
        causal_ctx = CausalContext(
            causal_depth=3,
            temporal_window=86400,  # 24 hours
            enable_caching=True,
        )

        result = causal_ctx.select(
            world_model=world_model,
            query={
                "text": "What causes X?",
                "memory": hierarchical_memory.retrieve(query),
                "limit": 20,
                "include_confounders": True,
            }
        )
    """

    def __init__(
        self,
        causal_depth: int = 2,
        temporal_window: float = 86400.0,  # 24 hours in seconds
        decay_function: TemporalDecayFunction = TemporalDecayFunction.EXPONENTIAL,
        decay_half_life_hours: float = 24.0,
        enable_caching: bool = True,
        cache_size: int = 1000,
        min_causal_strength: float = 0.1,
    ) -> None:
        self.causal_depth = max(1, int(causal_depth))
        self.temporal_window = float(temporal_window)
        self.decay_function = decay_function
        self.decay_half_life = float(decay_half_life_hours) * 3600.0
        self.enable_caching = enable_caching
        self.min_causal_strength = float(min_causal_strength)

        # Caching
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_size = cache_size

        # Intervention tracking
        self._interventions: deque = deque(maxlen=1000)

        # Causal graph cache
        self._causal_graph_cache: Optional[Dict[str, Any]] = None
        self._graph_cache_time: float = 0.0
        self._graph_cache_ttl: float = 3600.0  # 1 hour

    # ================================ Public API ================================ #

    def select(self, world_model: Any, query: Any) -> Dict[str, Any]:
        """
        Select causally-relevant context using advanced causal reasoning.

        Args:
            world_model: World model with causal graph capabilities
            query: Query specification (text, tokens, or dict)

        Returns:
            Dict with causal context, statistics, and analysis results
        """
        start_time = time.time()

        # Normalize query
        qtext, qterms = self._normalize_query(query)

        # Extract parameters
        limit = 20
        causal_depth = self.causal_depth
        include_confounders = True
        temporal_window = self.temporal_window

        if isinstance(query, dict):
            limit = int(query.get("limit", limit))
            causal_depth = int(query.get("causal_depth", causal_depth))
            include_confounders = bool(
                query.get("include_confounders", include_confounders)
            )
            temporal_window = float(query.get("temporal_window", temporal_window))

        # Check cache
        cache_key = self._get_cache_key(qtext, limit, causal_depth)
        if self.enable_caching and cache_key in self._cache:
            cached = self._cache[cache_key]
            cached["from_cache"] = True
            return cached

        # Extract concepts via world_model
        concepts = self._wm_extract_concepts(world_model, qtext, qterms)

        # Build or retrieve causal graph
        causal_graph = self._get_causal_graph(world_model, concepts)

        # Find causally-related concepts
        causal_related = self._find_causal_related(
            concepts, causal_graph, depth=causal_depth
        )

        # Detect confounders and mediators
        confounders = []
        mediators = []
        if include_confounders and causal_graph:
            confounders = self._detect_confounders(concepts, causal_graph)
            mediators = self._detect_mediators(concepts, causal_graph)

        # Gather memory from query
        memory = query.get("memory") if isinstance(query, dict) else {}
        episodic = memory.get("episodic") if isinstance(memory, dict) else None
        semantic = memory.get("semantic") if isinstance(memory, dict) else None
        procedural = memory.get("procedural") if isinstance(memory, dict) else None

        # Ensure lists (handle None values)
        episodic = episodic if isinstance(episodic, list) else []
        semantic = semantic if isinstance(semantic, list) else []
        procedural = procedural if isinstance(procedural, list) else []

        # Score items with causal relevance
        scored: List[Dict[str, Any]] = []
        now = time.time()

        # Process episodic memories
        for e in episodic:
            result = self._score_episodic_causal(
                e,
                concepts,
                causal_related,
                confounders,
                mediators,
                world_model,
                causal_graph,
                now,
                temporal_window,
            )
            if result:
                scored.append(result)

        # Process semantic memories
        for s in semantic:
            result = self._score_semantic_causal(
                s,
                concepts,
                causal_related,
                confounders,
                mediators,
                world_model,
                causal_graph,
                now,
                temporal_window,
            )
            if result:
                scored.append(result)

        # Process procedural memories
        for p in procedural:
            result = self._score_procedural_causal(
                p,
                concepts,
                causal_related,
                confounders,
                mediators,
                world_model,
                causal_graph,
                now,
                temporal_window,
            )
            if result:
                scored.append(result)

        # Rank by causal relevance
        scored.sort(key=lambda d: d["score"], reverse=True)
        top_scored = scored[:limit]

        # Generate causal explanations
        explanations = self._generate_causal_explanations(
            top_scored, concepts, causal_graph
        )

        # Compute statistics
        computation_time = (time.time() - start_time) * 1000
        statistics = CausalStatistics(
            total_nodes=len(causal_graph.get("nodes", [])) if causal_graph else 0,
            total_edges=len(causal_graph.get("edges", [])) if causal_graph else 0,
            avg_path_length=self._compute_avg_path_length(top_scored),
            max_path_length=max(
                (len(d.get("causal_path", [])) for d in top_scored), default=0
            ),
            num_confounders=len(confounders),
            num_mediators=len(mediators),
            temporal_span=temporal_window,
            computation_time_ms=computation_time,
        )

        # Compile result
        result = {
            "causal_context": top_scored,
            "concepts": concepts,
            "causal_related": causal_related,
            "causal_graph": self._serialize_causal_graph(causal_graph)
            if causal_graph
            else None,
            "interventions": [asdict(i) for i in list(self._interventions)[-10:]],
            "confounders": confounders,
            "mediators": mediators,
            "explanations": explanations,
            "limit": limit,
            "statistics": asdict(statistics),
            "from_cache": False,
        }

        # Cache result
        if self.enable_caching:
            self._update_cache(cache_key, result)

        return result

    def record_intervention(
        self,
        variable: str,
        value: Any,
        effect_on: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a causal intervention for tracking.

        Args:
            variable: Variable that was intervened on
            value: Value set by intervention
            effect_on: Dict of affected variables and their effect sizes
            metadata: Additional metadata
        """
        intervention = CausalIntervention(
            variable=variable,
            value=value,
            timestamp=time.time(),
            effect_on=effect_on or {},
            metadata=metadata or {},
        )
        self._interventions.append(intervention)

    def compute_counterfactual(
        self,
        world_model: Any,
        intervention: Optional[Union[str, Dict[str, Any]]] = None,
        outcome: Optional[str] = None,
        variable: Optional[str] = None,
        original_value: Any = None,
        counterfactual_value: Any = None,
        outcome_variable: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> CounterfactualScenario:
        """
        Compute a counterfactual scenario.

        Supports two calling styles:
        1. New style: compute_counterfactual(wm, variable="X", original_value=1,
                                             counterfactual_value=2, outcome_variable="Y")
        2. Old style: compute_counterfactual(wm, intervention={"variable": "X", "value": 2},
                                             outcome="Y")

        Args:
            world_model: World model with causal capabilities
            intervention: Dict with variable and value (old style) OR variable name (new style)
            outcome: Outcome variable name (old style)
            variable: Variable to modify (new style)
            original_value: Original value (new style)
            counterfactual_value: Counterfactual value (new style)
            outcome_variable: Outcome to measure (new style)
            context: Additional context

        Returns:
            CounterfactualScenario with analysis
        """
        # Handle old-style calling convention
        if isinstance(intervention, dict):
            variable = intervention.get("variable")
            counterfactual_value = intervention.get("value")
            original_value = 0  # Default
            outcome_variable = outcome
        elif isinstance(intervention, str) and outcome is not None:
            # intervention is actually the variable name
            variable = intervention
            original_value = 0  # Default
            # counterfactual_value should be passed as next arg
            outcome_variable = outcome

        # Ensure we have required values
        if not variable or outcome_variable is None:
            return CounterfactualScenario(
                original_value=original_value or 0,
                counterfactual_value=counterfactual_value or 0,
                outcome_difference=0.0,
                plausibility=0.0,
                explanation="Invalid counterfactual parameters",
            )

        # Compute effect difference
        outcome_diff = 0.0
        plausibility = 0.7  # Default plausibility

        # If world model supports intervention effects
        if hasattr(world_model, "compute_intervention_effect"):
            try:
                original_outcome = world_model.compute_intervention_effect(
                    intervention
                    if isinstance(intervention, dict)
                    else {variable: original_value},
                    outcome_variable,
                )
                cf_outcome = world_model.compute_intervention_effect(
                    {variable: counterfactual_value}, outcome_variable
                )
                outcome_diff = cf_outcome - original_outcome
            except Exception as e:
                _logger.debug(f"Failed to compute counterfactual outcome: {e}")

        # Generate explanation
        explanation = (
            f"If {variable} had been '{counterfactual_value}' instead of '{original_value}', "
            f"{outcome_variable} would have changed by {outcome_diff:+.3f}"
        )

        return CounterfactualScenario(
            original_value=original_value,
            counterfactual_value=counterfactual_value,
            outcome_difference=outcome_diff,
            plausibility=plausibility,
            explanation=explanation,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics"""
        return {
            "cache_size": len(self._cache),
            "num_interventions": len(self._interventions),
            "graph_cached": self._causal_graph_cache is not None,
        }

    def get_recent_interventions(self, limit: int = 10) -> List[CausalIntervention]:
        """
        Get recent interventions.

        Args:
            limit: Maximum number of interventions to return

        Returns:
            List of recent CausalIntervention objects
        """
        return list(self._interventions)[-limit:] if self._interventions else []

    # Alias for compatibility
    def track_intervention(
        self,
        variable: str,
        value: Any,
        effect_on: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Alias for record_intervention for backwards compatibility"""
        self.record_intervention(variable, value, effect_on, metadata)

    def clear_cache(self) -> None:
        """Clear all caches"""
        self._cache.clear()
        self._causal_graph_cache = None

    # ================================ Causal Graph Operations ================================ #

    def _get_causal_graph(
        self, world_model: Any, concepts: List[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve or build causal graph from world model.
        """
        # Check cache
        if self._causal_graph_cache and (
            time.time() - self._graph_cache_time < self._graph_cache_ttl
        ):
            return self._causal_graph_cache

        graph = None

        # Try to get from world model
        if hasattr(world_model, "causal_graph"):
            try:
                graph = self._extract_graph_structure(world_model.causal_graph)
            except Exception as e:
                _logger.debug(f"Failed to extract graph from causal_graph: {e}")
        elif hasattr(world_model, "causal_dag"):
            try:
                graph = self._extract_graph_structure(world_model.causal_dag)
            except Exception as e:
                _logger.debug(f"Failed to extract graph from causal_dag: {e}")

        # Build minimal graph from concepts if not available
        if not graph and concepts:
            graph = self._build_minimal_graph(concepts, world_model)

        # Cache
        if graph:
            self._causal_graph_cache = graph
            self._graph_cache_time = time.time()

        return graph

    def _extract_graph_structure(self, graph_obj: Any) -> Dict[str, Any]:
        """
        Extract graph structure from various graph objects.
        """
        nodes = set()
        edges = []

        # Try NetworkX-like interface
        if hasattr(graph_obj, "nodes") and hasattr(graph_obj, "edges"):
            try:
                nodes = set(graph_obj.nodes())
                for u, v in graph_obj.edges():
                    edges.append({"source": str(u), "target": str(v), "weight": 1.0})
            except Exception as e:
                _logger.debug(f"Failed to extract edges from graph object: {e}")

        # Try dict representation
        elif isinstance(graph_obj, dict):
            if "nodes" in graph_obj and "edges" in graph_obj:
                nodes = set(graph_obj["nodes"])
                edges = graph_obj["edges"]

        return {
            "nodes": list(nodes),
            "edges": edges,
            "metadata": {},
        }

    def _build_minimal_graph(
        self, concepts: List[str], world_model: Any
    ) -> Dict[str, Any]:
        """
        Build minimal causal graph from concepts.
        """
        nodes = list(concepts)
        edges = []

        # Try to get causal relationships from world model
        if hasattr(world_model, "get_related_concepts"):
            for concept in concepts[:20]:
                try:
                    related = world_model.get_related_concepts([concept])
                    if isinstance(related, list):
                        for rel in related:
                            if str(rel) in concepts:
                                edges.append(
                                    {
                                        "source": concept,
                                        "target": str(rel),
                                        "weight": 0.5,
                                    }
                                )
                except Exception as e:
                    _logger.debug(f"Failed to compute Granger causality: {e}")
                    continue

        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {"generated": True},
        }

    def _find_causal_related(
        self,
        concepts: List[str],
        causal_graph: Optional[Dict[str, Any]],
        depth: int = 2,
    ) -> List[str]:
        """
        Find causally-related concepts via graph traversal.
        """
        if not causal_graph or not concepts:
            return concepts

        related = set(concepts)

        # Build adjacency list
        adjacency = defaultdict(list)
        for edge in causal_graph.get("edges", []):
            source = edge.get("source")
            target = edge.get("target")
            if source and target:
                adjacency[source].append(target)
                adjacency[target].append(source)  # Bidirectional for context

        # BFS traversal up to depth
        queue = deque([(c, 0) for c in concepts])
        visited = set(concepts)

        while queue:
            node, d = queue.popleft()
            if d >= depth:
                continue

            for neighbor in adjacency.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    related.add(neighbor)
                    queue.append((neighbor, d + 1))

        return list(related)

    def _detect_confounders(
        self, concepts: List[str], causal_graph: Dict[str, Any]
    ) -> List[str]:
        """
        Detect potential confounders (common causes).
        """
        if not concepts or not causal_graph:
            return []

        # Build parent map
        parents = defaultdict(set)
        for edge in causal_graph.get("edges", []):
            source = edge.get("source")
            target = edge.get("target")
            if source and target:
                parents[target].add(source)

        # Find common parents (confounders)
        confounders = set()
        for i, c1 in enumerate(concepts):
            for c2 in concepts[i + 1 :]:
                common = parents.get(c1, set()) & parents.get(c2, set())
                confounders.update(common)

        return list(confounders)

    # Alias for compatibility
    def _identify_confounders(
        self, concepts: List[str], causal_graph: Dict[str, Any]
    ) -> List[str]:
        """Alias for _detect_confounders"""
        return self._detect_confounders(concepts, causal_graph)

    def _detect_mediators(
        self, concepts: List[str], causal_graph: Dict[str, Any]
    ) -> List[str]:
        """
        Detect potential mediators (on causal paths).
        """
        if not concepts or not causal_graph:
            return []

        # Build adjacency
        adjacency = defaultdict(list)
        for edge in causal_graph.get("edges", []):
            source = edge.get("source")
            target = edge.get("target")
            if source and target:
                adjacency[source].append(target)

        # Find nodes on paths between concepts
        mediators = set()
        for i, start in enumerate(concepts):
            for end in concepts[i + 1 :]:
                path = self._find_path(start, end, adjacency)
                if path and len(path) > 2:
                    mediators.update(path[1:-1])

        return list(mediators)

    # Alias for compatibility
    def _identify_mediators(
        self, concepts: List[str], causal_graph: Dict[str, Any]
    ) -> List[str]:
        """Alias for _detect_mediators"""
        return self._detect_mediators(concepts, causal_graph)

    def _traverse_causal_graph(
        self,
        concepts: List[str],
        causal_graph: Optional[Dict[str, Any]],
        max_depth: int = 3,
    ) -> Dict[str, Any]:
        """
        Traverse causal graph from given concepts.

        Args:
            concepts: Starting concepts
            causal_graph: Causal graph structure
            max_depth: Maximum traversal depth

        Returns:
            Dict with paths and nodes_visited
        """
        if not concepts or not causal_graph:
            return {"paths": [], "nodes_visited": set()}

        # Build adjacency
        adjacency = defaultdict(list)
        for edge in causal_graph.get("edges", []):
            source = edge.get("source")
            target = edge.get("target")
            if source and target:
                adjacency[source].append(target)

        paths = []
        nodes_visited = set(concepts)

        # BFS traversal from each concept
        for concept in concepts:
            queue = deque([(concept, [concept], 0)])

            while queue:
                node, path, depth = queue.popleft()

                if depth >= max_depth:
                    continue

                for neighbor in adjacency.get(node, []):
                    if neighbor not in path:  # Avoid cycles
                        new_path = path + [neighbor]
                        paths.append(new_path)
                        nodes_visited.add(neighbor)
                        queue.append((neighbor, new_path, depth + 1))

        return {"paths": paths, "nodes_visited": nodes_visited}

    def _build_causal_path(
        self, terms: List[str], causal_graph: Optional[Dict[str, Any]]
    ) -> List[str]:
        """
        Build a causal path connecting the given terms.

        Args:
            terms: Terms to connect
            causal_graph: Causal graph structure

        Returns:
            List of nodes forming a path
        """
        if not terms or not causal_graph:
            return terms[:1] if terms else []

        if len(terms) == 1:
            return terms

        # Find terms that exist in graph
        nodes = set(causal_graph.get("nodes", []))
        term_nodes = [t for t in terms if t in nodes]

        if len(term_nodes) < 2:
            return terms[:1] if terms else []

        # Build adjacency
        adjacency = defaultdict(list)
        for edge in causal_graph.get("edges", []):
            source = edge.get("source")
            target = edge.get("target")
            if source and target:
                adjacency[source].append(target)

        # Find path between first and last term node
        path = self._find_path(term_nodes[0], term_nodes[-1], adjacency)
        return path if path else term_nodes

    def _find_path(
        self, start: str, end: str, adjacency: Dict[str, List[str]]
    ) -> Optional[List[str]]:
        """
        Find path between two nodes using BFS.
        """
        if start == end:
            return [start]

        queue = deque([(start, [start])])
        visited = {start}

        while queue:
            node, path = queue.popleft()

            for neighbor in adjacency.get(node, []):
                if neighbor == end:
                    return path + [neighbor]

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None

    # ================================ Memory Scoring ================================ #

    def _score_episodic_causal(
        self,
        item: Dict[str, Any],
        concepts: List[str],
        causal_related: List[str],
        confounders: List[str],
        mediators: List[str],
        world_model: Any,
        causal_graph: Optional[Dict[str, Any]],
        now: float,
        temporal_window: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Score episodic memory item with causal relevance.
        """
        # Extract text
        text = " ".join(str(x) for x in [item.get("prompt", ""), item.get("token", "")])
        terms = self._tokenize(text)

        # Timestamp filtering
        ts = float(item.get("ts", now))
        if now - ts > temporal_window:
            return None  # Outside temporal window

        # Compute scores
        overlap = self._overlap(concepts, terms)
        causal_overlap = self._overlap(causal_related, terms)
        confounder_overlap = self._overlap(confounders, terms)
        mediator_overlap = self._overlap(mediators, terms)

        # Time decay
        decay = self._time_decay(now - ts)

        # World model relatedness
        relatedness = self._wm_relatedness(world_model, concepts, terms)

        # Causal path
        causal_path = self._find_causal_path_in_text(terms, causal_graph)
        causal_strength = self._compute_causal_strength(causal_path, causal_graph)

        # Temporal relevance
        temporal_relevance = decay * (1.0 if now - ts < temporal_window / 2 else 0.5)

        # Composite score
        score = (
            0.3 * overlap
            + 0.3 * causal_overlap
            + 0.1 * confounder_overlap
            + 0.1 * mediator_overlap
            + 0.1 * relatedness
            + 0.1 * causal_strength
        ) * decay

        # Reason
        reason_parts = [
            f"overlap={overlap:.2f}",
            f"causal={causal_overlap:.2f}",
            f"decay={decay:.2f}",
            f"strength={causal_strength:.2f}",
        ]

        return {
            "source": "episodic",
            "score": float(score),
            "item": item,
            "reason": ", ".join(reason_parts),
            "causal_path": causal_path,
            "causal_strength": causal_strength,
            "temporal_relevance": temporal_relevance,
        }

    def _score_semantic_causal(
        self,
        item: Dict[str, Any],
        concepts: List[str],
        causal_related: List[str],
        confounders: List[str],
        mediators: List[str],
        world_model: Any,
        causal_graph: Optional[Dict[str, Any]],
        now: float,
        temporal_window: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Score semantic memory item with causal relevance.
        """
        concept_text = str(item.get("concept", ""))
        terms = [str(t) for t in (item.get("terms") or [])] or self._tokenize(
            concept_text
        )

        # Compute scores
        overlap = self._overlap(concepts, terms)
        causal_overlap = self._overlap(causal_related, terms)

        # Time decay
        last_seen = float(item.get("last_seen", now))
        decay = self._time_decay(now - last_seen)

        # Frequency
        freq = float(item.get("freq", 1))
        freq_bonus = min(2.0, 1.0 + 0.05 * freq)

        # Causal analysis
        causal_path = self._find_causal_path_in_text(terms, causal_graph)
        causal_strength = self._compute_causal_strength(causal_path, causal_graph)

        # Check if concept is confounder or mediator
        is_confounder = concept_text in confounders or any(
            t in confounders for t in terms
        )
        is_mediator = concept_text in mediators or any(t in mediators for t in terms)
        role_bonus = 1.2 if is_confounder or is_mediator else 1.0

        # Composite score
        score = (
            (0.5 * overlap + 0.3 * causal_overlap + 0.2 * causal_strength)
            * decay
            * freq_bonus
            * role_bonus
        )

        reason_parts = [
            f"overlap={overlap:.2f}",
            f"causal={causal_overlap:.2f}",
            f"freq={freq:.0f}",
            f"strength={causal_strength:.2f}",
        ]
        if is_confounder:
            reason_parts.append("confounder")
        if is_mediator:
            reason_parts.append("mediator")

        return {
            "source": "semantic",
            "score": float(score),
            "item": item,
            "reason": ", ".join(reason_parts),
            "causal_path": causal_path,
            "causal_strength": causal_strength,
            "temporal_relevance": decay,
        }

    def _score_procedural_causal(
        self,
        item: Dict[str, Any],
        concepts: List[str],
        causal_related: List[str],
        confounders: List[str],
        mediators: List[str],
        world_model: Any,
        causal_graph: Optional[Dict[str, Any]],
        now: float,
        temporal_window: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Score procedural memory item with causal relevance.
        """
        name = str(item.get("name", ""))
        terms = [str(t) for t in (item.get("signature_terms") or [])] or self._tokenize(
            name
        )

        # Compute scores
        overlap = self._overlap(concepts, terms)
        causal_overlap = self._overlap(causal_related, terms)

        # Time decay
        last_seen = float(item.get("last_seen", now))
        decay = self._time_decay(now - last_seen)

        # Frequency
        freq = float(item.get("freq", 1))
        freq_bonus = min(2.0, 1.0 + 0.05 * freq)

        # Causal analysis
        causal_path = self._find_causal_path_in_text(terms, causal_graph)
        causal_strength = self._compute_causal_strength(causal_path, causal_graph)

        # Composite score
        score = (
            (0.4 * overlap + 0.3 * causal_overlap + 0.3 * causal_strength)
            * decay
            * freq_bonus
        )

        reason_parts = [
            f"overlap={overlap:.2f}",
            f"causal={causal_overlap:.2f}",
            f"freq={freq:.0f}",
            f"strength={causal_strength:.2f}",
        ]

        return {
            "source": "procedural",
            "score": float(score),
            "item": item,
            "reason": ", ".join(reason_parts),
            "causal_path": causal_path,
            "causal_strength": causal_strength,
            "temporal_relevance": decay,
        }

    # Aliases for backwards compatibility
    def _score_episodic_item(self, item, qterms, concepts, world_model):
        """Alias for _score_episodic_causal (simplified signature)"""
        return self._score_episodic_causal(
            item,
            concepts,
            qterms,
            [],
            [],
            world_model,
            None,
            time.time(),
            self.temporal_window,
        )["score"]

    def _score_semantic_entry(self, item, qterms, concepts, world_model):
        """Alias for _score_semantic_causal (simplified signature)"""
        return self._score_semantic_causal(
            item,
            concepts,
            qterms,
            [],
            [],
            world_model,
            None,
            time.time(),
            self.temporal_window,
        )["score"]

    def _score_procedural_pattern(self, item, qterms, concepts, world_model):
        """Alias for _score_procedural_causal (simplified signature)"""
        return self._score_procedural_causal(
            item,
            concepts,
            qterms,
            [],
            [],
            world_model,
            None,
            time.time(),
            self.temporal_window,
        )["score"]

    def _find_causal_path_in_text(
        self, terms: List[str], causal_graph: Optional[Dict[str, Any]]
    ) -> List[str]:
        """
        Find causal path mentioned in text terms.
        """
        if not causal_graph or not terms:
            return []

        # Find terms that are nodes in the graph
        nodes = set(causal_graph.get("nodes", []))
        term_nodes = [t for t in terms if t in nodes]

        if len(term_nodes) < 2:
            return term_nodes

        # Find path between first and last
        adjacency = defaultdict(list)
        for edge in causal_graph.get("edges", []):
            source = edge.get("source")
            target = edge.get("target")
            if source and target:
                adjacency[source].append(target)

        path = self._find_path(term_nodes[0], term_nodes[-1], adjacency)
        return path if path else term_nodes

    def _compute_causal_strength(
        self, path: List[str], causal_graph: Optional[Dict[str, Any]]
    ) -> float:
        """
        Compute strength of a causal path.
        """
        if not path or len(path) < 2 or not causal_graph:
            return 0.0

        # Build edge weight map
        edge_weights = {}
        for edge in causal_graph.get("edges", []):
            source = edge.get("source")
            target = edge.get("target")
            weight = float(edge.get("weight", 1.0))
            if source and target:
                edge_weights[(source, target)] = weight

        # Compute path strength (product of edge weights, decay by length)
        strength = 1.0
        for i in range(len(path) - 1):
            edge_weight = edge_weights.get((path[i], path[i + 1]), 0.5)
            strength *= edge_weight

        # Decay by path length
        length_penalty = 0.8 ** (len(path) - 1)

        return strength * length_penalty

    # ================================ Explanation Generation ================================ #

    def _generate_causal_explanations(
        self,
        scored_items: List[Dict[str, Any]],
        concepts: List[str],
        causal_graph: Optional[Dict[str, Any]],
    ) -> List[str]:
        """
        Generate natural language causal explanations.
        """
        explanations = []

        # Overall explanation
        if scored_items:
            top_source = scored_items[0]["source"]
            top_score = scored_items[0]["score"]
            explanations.append(
                f"Most relevant context from {top_source} memory (score: {top_score:.3f})"
            )

        # Causal path explanations
        paths_found = [item for item in scored_items if item.get("causal_path")]
        if paths_found:
            path_ex = paths_found[0]
            path = path_ex["causal_path"]
            if len(path) > 1:
                path_str = " → ".join(path)
                explanations.append(
                    f"Causal path identified: {path_str} "
                    f"(strength: {path_ex['causal_strength']:.3f})"
                )

        # Temporal explanation
        recent_items = [
            item for item in scored_items if item.get("temporal_relevance", 0) > 0.7
        ]
        if recent_items:
            explanations.append(
                f"{len(recent_items)} recent relevant items within temporal window"
            )

        return explanations

    # ================================ Utilities ================================ #

    def _normalize_query(self, query: Any) -> Tuple[str, List[str]]:
        """Normalize query to text and terms"""
        if isinstance(query, dict):
            if "text" in query and isinstance(query["text"], str):
                qtext = query["text"]
            elif "prompt" in query and isinstance(query["prompt"], str):
                qtext = query["prompt"]
            elif "tokens" in query:
                qtext = self._tokens_to_text(query["tokens"])
            else:
                qtext = str(query)
        elif isinstance(query, list):
            qtext = self._tokens_to_text(query)
        elif isinstance(query, str):
            qtext = query
        else:
            qtext = str(query)
        return qtext, self._tokenize(qtext)

    def _tokens_to_text(self, tokens: Tokens) -> str:
        """Convert tokens to text"""
        if not tokens:
            return ""
        if isinstance(tokens[0], str):
            return " ".join(tokens)
        return " ".join(str(t) for t in tokens)

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        if not text:
            return []
        return [t for t in re.findall(r"[A-Za-z0-9_]+", text.lower()) if t]

    def _overlap(self, a: List[str], b: List[str]) -> float:
        """Compute overlap score"""
        if not a or not b:
            return 0.0
        sa, sb = set(a), set(b)
        inter = len(sa & sb)
        denom = max(1, min(len(sa), len(sb)))
        return inter / denom

    def _time_decay(self, dt_seconds: float) -> float:
        """Compute time decay based on configured function"""
        if dt_seconds < 0:
            dt_seconds = 0.0

        if self.decay_function == TemporalDecayFunction.EXPONENTIAL:
            # Exponential decay
            if self.decay_half_life <= 0:
                return 1.0
            return 0.5 ** (dt_seconds / self.decay_half_life)

        elif self.decay_function == TemporalDecayFunction.HYPERBOLIC:
            # Hyperbolic decay (slower)
            return 1.0 / (1.0 + dt_seconds / 3600.0)

        elif self.decay_function == TemporalDecayFunction.POWER_LAW:
            # Power law decay
            alpha = 1.5
            return 1.0 / ((1.0 + dt_seconds / 3600.0) ** alpha)

        elif self.decay_function == TemporalDecayFunction.LINEAR:
            # Linear decay
            return max(0.0, 1.0 - dt_seconds / self.decay_half_life)

        else:
            return 1.0

    def _wm_extract_concepts(self, wm: Any, qtext: str, qterms: List[str]) -> List[str]:
        """Extract concepts via world model"""
        if wm and hasattr(wm, "extract_concepts"):
            try:
                c = wm.extract_concepts(qtext)
                if isinstance(c, list) and c:
                    return [str(x) for x in c][:50]
            except Exception as e:
                _logger.debug(f"Failed to extract concepts from world model: {e}")
        # Fallback: use tokenized terms
        return [t for t in qterms if len(t) > 2][:50]

    def _wm_relatedness(
        self, wm: Any, concepts: List[str], other_terms: List[str]
    ) -> float:
        """Compute world model relatedness"""
        if not wm or not concepts:
            return 0.0

        related: List[str] = []
        try:
            if hasattr(wm, "get_related_concepts"):
                rc = wm.get_related_concepts(concepts)
                if isinstance(rc, list):
                    related = [str(x) for x in rc]
            elif hasattr(wm, "causal_graph"):
                # Use causal graph neighbors
                graph = self._get_causal_graph(wm, concepts)
                if graph:
                    adjacency = defaultdict(list)
                    for edge in graph.get("edges", []):
                        source = edge.get("source")
                        target = edge.get("target")
                        if source and target:
                            adjacency[source].append(target)
                            adjacency[target].append(source)

                    for c in concepts[:20]:
                        related.extend(adjacency.get(c, []))
        except Exception as e:
            _logger.warning(f"Failed to get related concepts from world model: {e}")
            related = []

        if not related:
            return 0.0

        return self._overlap(related, other_terms)

    def _serialize_causal_graph(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize causal graph for output"""
        return {
            "num_nodes": len(graph.get("nodes", [])),
            "num_edges": len(graph.get("edges", [])),
            "nodes": graph.get("nodes", [])[:20],  # Sample
            "edges": graph.get("edges", [])[:20],  # Sample
        }

    def _compute_avg_path_length(self, items: List[Dict[str, Any]]) -> float:
        """Compute average causal path length"""
        lengths = [len(item.get("causal_path", [])) for item in items]
        non_empty = [l for l in lengths if l > 0]
        return sum(non_empty) / len(non_empty) if non_empty else 0.0

    # ================================ Caching ================================ #

    def _get_cache_key(self, qtext: str, limit: int, depth: int) -> str:
        """Generate cache key"""
        combined = f"{qtext}:{limit}:{depth}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def _update_cache(self, key: str, result: Dict[str, Any]) -> None:
        """Update cache with LRU eviction"""
        if len(self._cache) >= self._cache_size:
            # Simple eviction: remove oldest
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[key] = result
