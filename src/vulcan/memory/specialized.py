"""Specialized memory types: Episodic, Semantic, Procedural, Working"""

import copy
import hashlib
import json
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from .base import (
    BaseMemorySystem,
    Memory,
    MemoryConfig,
    MemoryQuery,
    MemoryType,
    RetrievalResult,
)
from .hierarchical import HierarchicalMemory

# Try to import optional dependencies
try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

# Try to import VULCAN for evolution engine integration
try:
    from unified_runtime_core import get_runtime

    VULCAN_AVAILABLE = True
except ImportError:
    VULCAN_AVAILABLE = False

logger = logging.getLogger(__name__)

# ============================================================
# EVOLUTION ENGINE WITH VULCAN INTEGRATION
# ============================================================


@dataclass
class Individual:
    """Individual in evolution population."""

    graph: Dict[str, Any]
    fitness: float = 0.0
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    mutations: List[str] = field(default_factory=list)


class EvolutionEngine:
    """Evolution engine with optional VULCAN fitness integration."""

    def __init__(self, population_size: int = 50, use_vulcan_fitness: bool = True):
        self.population_size = population_size
        self.use_vulcan_fitness = use_vulcan_fitness
        self.population: List[Individual] = []
        self.generation = 0
        self.fitness_cache = {}
        self.best_individual: Optional[Individual] = None
        self.evolution_history = []

    def _hash_graph(self, graph: Dict) -> str:
        """Create hash of graph for caching."""
        graph_str = json.dumps(graph, sort_keys=True, default=str)
        return hashlib.sha256(graph_str.encode()).hexdigest()

    def _validate_graph(self, graph: Dict) -> Dict:
        """Validate and normalize graph structure."""
        validated = {
            "nodes": graph.get("nodes", []),
            "edges": graph.get("edges", []),
            "metadata": graph.get("metadata", {}),
        }
        return validated

    def _vulcan_fitness_component(self, graph: Dict) -> float:
        """
        Optional: Get VULCAN's assessment as part of fitness.

        Returns:
            Fitness component from VULCAN (0-1)
        """
        if not VULCAN_AVAILABLE:
            return 0.5  # Neutral

        try:
            runtime = get_runtime()
            if not hasattr(runtime, "vulcan_bridge") or not runtime.vulcan_bridge:
                return 0.5

            # Get VULCAN's evaluation
            evaluation = runtime.vulcan_bridge.world_model.evaluate_graph_proposal(
                {"graph": graph, "timestamp": datetime.utcnow().isoformat()}
            )

            # Convert VULCAN assessment to fitness component
            if not evaluation.get("valid", True):
                return 0.0  # Penalize invalid graphs

            safety_level = evaluation.get("safety_level", 0.5)
            return min(1.0, max(0.0, safety_level))

        except Exception as e:
            logger.debug(f"VULCAN fitness component unavailable: {e}")
            return 0.5  # Neutral on error

    def _evaluate_population(self, fitness_function: Callable):
        """Synchronous fitness evaluation with optional VULCAN component."""
        for individual in self.population:
            graph_hash = self._hash_graph(individual.graph)
            cached_fitness = self.fitness_cache.get(graph_hash)

            if cached_fitness is not None:
                individual.fitness = cached_fitness
            else:
                try:
                    validated = self._validate_graph(individual.graph)

                    # Base fitness
                    base_fitness = fitness_function(validated)

                    # Optional: Add VULCAN component
                    if VULCAN_AVAILABLE and self.use_vulcan_fitness:
                        vulcan_component = self._vulcan_fitness_component(validated)
                        # Weighted combination
                        individual.fitness = 0.7 * base_fitness + 0.3 * vulcan_component
                    else:
                        individual.fitness = base_fitness

                    # Clamp and cache
                    individual.fitness = max(0.0, min(1.0, individual.fitness))
                    self.fitness_cache[graph_hash] = individual.fitness

                except Exception as e:
                    logger.warning(f"Fitness evaluation failed: {e}")
                    individual.fitness = 0.0

    def evolve(self, fitness_function: Callable, generations: int = 100):
        """
        Evolve population over multiple generations.

        Args:
            fitness_function: Function to evaluate individual fitness
            generations: Number of generations to evolve
        """
        for gen in range(generations):
            self.generation = gen

            # Evaluate fitness
            self._evaluate_population(fitness_function)

            # Track best individual
            best = max(self.population, key=lambda x: x.fitness)
            if (
                self.best_individual is None
                or best.fitness > self.best_individual.fitness
            ):
                self.best_individual = copy.deepcopy(best)

            # Record history
            self.evolution_history.append(
                {
                    "generation": gen,
                    "best_fitness": best.fitness,
                    "avg_fitness": np.mean([ind.fitness for ind in self.population]),
                    "diversity": self._calculate_diversity(),
                }
            )

            # Selection and reproduction
            self._selection_and_reproduction()

            logger.info(f"Generation {gen}: Best fitness = {best.fitness:.4f}")

    def _calculate_diversity(self) -> float:
        """Calculate population diversity."""
        if len(self.population) < 2:
            return 0.0

        hashes = [self._hash_graph(ind.graph) for ind in self.population]
        unique_hashes = len(set(hashes))
        return unique_hashes / len(hashes)

    def _selection_and_reproduction(self):
        """Select parents and create next generation."""
        # Tournament selection
        new_population = []

        while len(new_population) < self.population_size:
            # Select parents
            parent1 = self._tournament_select()
            parent2 = self._tournament_select()

            # Crossover
            child = self._crossover(parent1, parent2)

            # Mutation
            child = self._mutate(child)

            new_population.append(child)

        self.population = new_population

    def _tournament_select(self, tournament_size: int = 3) -> Individual:
        """Tournament selection."""
        tournament = np.random.choice(
            self.population, size=tournament_size, replace=False
        )
        return max(tournament, key=lambda x: x.fitness)

    def _crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """Crossover two parents to create offspring."""
        child_graph = {}

        # Combine nodes
        nodes1 = set(parent1.graph.get("nodes", []))
        nodes2 = set(parent2.graph.get("nodes", []))
        child_graph["nodes"] = list(nodes1 | nodes2)

        # Combine edges (randomly select from parents)
        edges1 = parent1.graph.get("edges", [])
        edges2 = parent2.graph.get("edges", [])

        child_graph["edges"] = []
        for edge in edges1 + edges2:
            if np.random.random() < 0.5:
                child_graph["edges"].append(edge)

        # Metadata from fitter parent
        if parent1.fitness > parent2.fitness:
            child_graph["metadata"] = copy.deepcopy(parent1.graph.get("metadata", {}))
        else:
            child_graph["metadata"] = copy.deepcopy(parent2.graph.get("metadata", {}))

        child = Individual(
            graph=child_graph,
            generation=self.generation + 1,
            parent_ids=[
                self._hash_graph(parent1.graph),
                self._hash_graph(parent2.graph),
            ],
        )

        return child

    def _mutate(self, individual: Individual, mutation_rate: float = 0.1) -> Individual:
        """Mutate individual."""
        if np.random.random() < mutation_rate:
            # Random mutation type
            mutation_type = np.random.choice(
                ["add_node", "remove_node", "add_edge", "remove_edge"]
            )

            if mutation_type == "add_node":
                new_node = f"node_{np.random.randint(1000)}"
                if "nodes" not in individual.graph:
                    individual.graph["nodes"] = []
                individual.graph["nodes"].append(new_node)
                individual.mutations.append("add_node")

            elif mutation_type == "remove_node" and individual.graph.get("nodes"):
                if len(individual.graph["nodes"]) > 1:
                    individual.graph["nodes"].pop(
                        np.random.randint(len(individual.graph["nodes"]))
                    )
                    individual.mutations.append("remove_node")

            elif mutation_type == "add_edge":
                if individual.graph.get("nodes") and len(individual.graph["nodes"]) > 1:
                    nodes = individual.graph["nodes"]
                    edge = (np.random.choice(nodes), np.random.choice(nodes))
                    if "edges" not in individual.graph:
                        individual.graph["edges"] = []
                    individual.graph["edges"].append(edge)
                    individual.mutations.append("add_edge")

            elif mutation_type == "remove_edge" and individual.graph.get("edges"):
                if len(individual.graph["edges"]) > 0:
                    individual.graph["edges"].pop(
                        np.random.randint(len(individual.graph["edges"]))
                    )
                    individual.mutations.append("remove_edge")

        return individual


# ============================================================
# EPISODIC MEMORY
# ============================================================


@dataclass
class Episode:
    """Single episode in episodic memory."""

    id: str
    start_time: float
    end_time: Optional[float]
    events: List[Dict[str, Any]]
    context: Dict[str, Any]
    outcome: Optional[Any] = None
    value: float = 0.0
    emotional_valence: float = 0.0  # -1 to 1
    tags: Set[str] = field(default_factory=set)
    embedding: Optional[np.ndarray] = None
    importance: float = 0.5

    def add_event(self, event: Dict[str, Any]):
        """Add event to episode."""
        self.events.append(
            {"timestamp": time.time(), "sequence": len(self.events), **event}
        )

        # Update importance based on event significance
        if "importance" in event:
            self.importance = max(self.importance, event["importance"])

    def compute_duration(self) -> float:
        """Compute episode duration."""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    def get_summary(self) -> Dict[str, Any]:
        """Get episode summary."""
        return {
            "id": self.id,
            "duration": self.compute_duration(),
            "num_events": len(self.events),
            "outcome": self.outcome,
            "value": self.value,
            "context_keys": list(self.context.keys()),
            "tags": list(self.tags),
        }

    def to_memory(self) -> Memory:
        """Convert episode to Memory object."""
        return Memory(
            id=self.id,
            type=MemoryType.EPISODIC,
            content={
                "events": self.events,
                "context": self.context,
                "outcome": self.outcome,
                "value": self.value,
            },
            embedding=self.embedding,
            timestamp=self.start_time,
            importance=self.importance,
            metadata={
                "episode_id": self.id,
                "duration": self.compute_duration(),
                "emotional_valence": self.emotional_valence,
                "tags": list(self.tags),
            },
        )


class EpisodicMemory(BaseMemorySystem):
    """Episodic memory for storing experiences and events."""

    def __init__(self, config: MemoryConfig):
        super().__init__(config)
        self.episodes: Dict[str, Episode] = {}
        self.current_episode: Optional[Episode] = None
        self.episode_index = defaultdict(set)  # key -> episode_ids
        self.temporal_index = []  # Sorted list of (timestamp, episode_id)
        self.tag_index = defaultdict(set)  # tag -> episode_ids

        # Episode chains for sequential patterns
        self.episode_chains = defaultdict(list)  # pattern_hash -> [episode_ids]
        
        # FIX #MEM-6: Add lock for thread-safe episode operations
        self._episode_lock = threading.RLock()

        # Hierarchical storage - ensure config has required attributes
        # Create a copy with safe attribute access
        try:
            self.hierarchical_memory = HierarchicalMemory(config)
        except AttributeError as e:
            logger.warning(f"Config missing attributes for HierarchicalMemory: {e}")

            # Create minimal config with required attributes
            class MinimalConfig:
                def __init__(self, base_config):
                    self.max_working_memory = getattr(
                        base_config, "max_working_memory", 20
                    )
                    self.max_short_term = getattr(base_config, "max_short_term", 1000)
                    self.max_long_term = getattr(base_config, "max_long_term", 100000)
                    self.consolidation_interval = getattr(
                        base_config, "consolidation_interval", 1000
                    )

            self.hierarchical_memory = HierarchicalMemory(MinimalConfig(config))

        self._init_indices()

    def _init_indices(self):
        """Initialize search indices."""
        self.embedding_index = {}  # episode_id -> embedding
        self.value_index = []  # Sorted list of (value, episode_id)

    def start_episode(self, context: Dict[str, Any], tags: Set[str] = None) -> str:
        """Start new episode."""
        # End current episode if exists
        if self.current_episode:
            self.end_episode()

        episode_id = f"episode_{time.time()}_{hashlib.sha256(str(context).encode()).hexdigest()[:8]}"

        self.current_episode = Episode(
            id=episode_id,
            start_time=time.time(),
            end_time=None,
            events=[],
            context=context,
            tags=tags or set(),
        )

        self.episodes[episode_id] = self.current_episode

        # Index by context
        self._index_episode(self.current_episode)

        return episode_id

    def end_episode(
        self, outcome: Any = None, value: float = 0.0, emotional_valence: float = 0.0
    ):
        """End current episode."""
        # FIX #MEM-6: Use lock for thread-safe episode access
        with self._episode_lock:
            if self.current_episode is None:
                return
            
            # Create a local reference and clear atomically while holding lock
            episode = self.current_episode
            self.current_episode = None

        # Process episode outside the lock (these operations are episode-local)
        episode.end_time = time.time()
        episode.outcome = outcome
        episode.value = value
        episode.emotional_valence = emotional_valence

        # Calculate importance
        episode.importance = self._calculate_episode_importance(episode)

        # Generate embedding
        episode.embedding = self._generate_episode_embedding(episode)

        # Update indices
        self._update_indices(episode)

        # Store in hierarchical memory
        memory = episode.to_memory()
        self.hierarchical_memory.store(
            memory.content,
            memory_type=MemoryType.EPISODIC,
            importance=memory.importance,
        )

        # Detect patterns
        self._detect_episode_patterns(episode)

    def add_event(self, event: Dict[str, Any]):
        """Add event to current episode."""
        # FIX #MEM-6: Use lock to prevent TOCTOU race condition
        # Without lock, end_episode() could set current_episode to None
        # between our check and use, causing AttributeError
        with self._episode_lock:
            episode = self.current_episode
            if episode is None:
                logger.debug("Cannot add event: no active episode")
                return
            
            episode.add_event(event)
            
            # Check if auto-end needed (still holding episode reference)
            should_end = len(episode.events) > 100
        
        # Auto-end episode if too long (outside lock to prevent deadlock)
        if should_end:
            self.end_episode()

    def store(self, content: Any, **kwargs) -> Memory:
        """Store content as episodic memory."""
        # Create episode from content
        episode_id = f"episode_{time.time()}_{id(content)}"

        episode = Episode(
            id=episode_id,
            start_time=time.time(),
            end_time=time.time(),
            events=[{"content": content, "timestamp": time.time()}],
            context=kwargs.get("context", {}),
            importance=kwargs.get("importance", 0.5),
        )

        # Generate embedding
        episode.embedding = self._generate_episode_embedding(episode)

        # Store episode
        self.episodes[episode_id] = episode
        self._index_episode(episode)
        self._update_indices(episode)

        # Create and return memory
        memory = episode.to_memory()

        # Also store in hierarchical memory
        self.hierarchical_memory.store(
            memory.content,
            memory_type=MemoryType.EPISODIC,
            importance=memory.importance,
        )

        self.stats.total_stores += 1

        return memory

    def retrieve(self, query: MemoryQuery) -> RetrievalResult:
        """Retrieve episodic memories."""
        start_time = time.time()

        # Convert episodes to memories
        episode_memories = [ep.to_memory() for ep in self.episodes.values()]
        memories_dict = {m.id: m for m in episode_memories}

        # Use hierarchical memory's retrieval
        result = self.hierarchical_memory.retrieve(query)

        # Enhance with episode-specific search
        if query.content and isinstance(query.content, dict):
            if "context" in query.content:
                # Find similar episodes
                similar = self.recall_similar_episodes(
                    query.content["context"], limit=query.limit
                )

                # Merge results
                episode_results = [
                    (
                        ep.to_memory(),
                        self._compute_context_similarity(
                            query.content["context"], ep.context
                        ),
                    )
                    for ep in similar
                ]

                # Combine with existing results
                combined = list(result.memories)
                combined_scores = list(result.scores)

                for mem, score in episode_results:
                    if mem.id not in [m.id for m in combined]:
                        combined.append(mem)
                        combined_scores.append(score)

                # Sort and limit
                sorted_pairs = sorted(
                    zip(combined, combined_scores), key=lambda x: x[1], reverse=True
                )[: query.limit]

                result.memories = [m for m, _ in sorted_pairs]
                result.scores = [s for _, s in sorted_pairs]

        result.query_time_ms = (time.time() - start_time) * 1000
        self.stats.total_queries += 1

        return result

    def forget(self, memory_id: str) -> bool:
        """Remove episodic memory."""
        # Remove from episodes
        if memory_id in self.episodes:
            episode = self.episodes[memory_id]

            # Remove from indices
            for key in episode.context.keys():
                self.episode_index[key].discard(memory_id)

            for tag in episode.tags:
                self.tag_index[tag].discard(memory_id)

            if memory_id in self.embedding_index:
                del self.embedding_index[memory_id]

            del self.episodes[memory_id]

            # Also remove from hierarchical memory
            self.hierarchical_memory.forget(memory_id)

            self.stats.total_memories -= 1
            return True

        return False

    def consolidate(self) -> int:
        """Consolidate episodic memories."""
        # Consolidate old episodes
        current_time = time.time()
        consolidation_threshold = 7 * 24 * 3600  # 7 days

        old_episodes = [
            ep
            for ep in self.episodes.values()
            if current_time - ep.start_time > consolidation_threshold
        ]

        if not old_episodes:
            return 0

        # Group similar episodes
        groups = self._group_similar_episodes(old_episodes)

        consolidated_count = 0

        for group in groups:
            if len(group) > 1:
                # Merge episodes
                merged = self._merge_episodes(group)

                # Remove old episodes
                for ep in group:
                    if ep.id != merged.id:
                        self.forget(ep.id)
                        consolidated_count += 1

                # Store merged episode
                self.episodes[merged.id] = merged
                self._index_episode(merged)
                self._update_indices(merged)

        # Also consolidate hierarchical memory
        consolidated_count += self.hierarchical_memory.consolidate()

        self.stats.total_consolidations += 1

        return consolidated_count

    def recall_similar_episodes(
        self, context: Dict[str, Any], limit: int = 5
    ) -> List[Episode]:
        """Recall episodes similar to context."""
        similarities = []

        for episode in self.episodes.values():
            similarity = self._compute_context_similarity(context, episode.context)

            # Boost recent episodes
            recency_boost = np.exp(
                -(time.time() - episode.start_time) / (7 * 24 * 3600)
            )
            adjusted_similarity = similarity * (1 + 0.2 * recency_boost)

            similarities.append((episode, adjusted_similarity))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        return [ep for ep, _ in similarities[:limit]]

    def recall_by_tags(self, tags: Set[str], limit: int = 10) -> List[Episode]:
        """Recall episodes by tags."""
        episode_ids = set()

        for tag in tags:
            if tag in self.tag_index:
                episode_ids.update(self.tag_index[tag])

        episodes = [self.episodes[eid] for eid in episode_ids if eid in self.episodes]

        # Sort by recency
        episodes.sort(key=lambda ep: ep.start_time, reverse=True)

        return episodes[:limit]

    def recall_by_value_range(
        self, min_value: float, max_value: float
    ) -> List[Episode]:
        """Recall episodes within value range."""
        return [
            ep for ep in self.episodes.values() if min_value <= ep.value <= max_value
        ]

    def find_patterns(self, min_support: int = 2) -> List[List[str]]:
        """Find recurring episode patterns."""
        patterns = []

        for pattern_hash, episode_ids in self.episode_chains.items():
            if len(episode_ids) >= min_support:
                patterns.append(episode_ids)

        return patterns

    def _compute_context_similarity(self, context1: Dict, context2: Dict) -> float:
        """Compute similarity between contexts."""
        if not context1 or not context2:
            return 0.0

        # Key overlap
        keys1 = set(context1.keys())
        keys2 = set(context2.keys())

        if not keys1 and not keys2:
            return 1.0
        if not keys1 or not keys2:
            return 0.0

        key_overlap = len(keys1 & keys2) / len(keys1 | keys2)

        # Value similarity for shared keys
        shared_keys = keys1 & keys2
        value_similarity = 0.0

        for key in shared_keys:
            val1 = context1[key]
            val2 = context2[key]

            if val1 == val2:
                value_similarity += 1.0
            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numeric similarity
                diff = abs(val1 - val2)
                max_val = max(abs(val1), abs(val2), 1)
                value_similarity += max(0, 1 - diff / max_val)
            elif isinstance(val1, str) and isinstance(val2, str):
                # String similarity
                value_similarity += self._string_similarity(val1, val2)

        if shared_keys:
            value_similarity /= len(shared_keys)

        # Weighted combination
        return 0.6 * key_overlap + 0.4 * value_similarity

    def _string_similarity(self, s1: str, s2: str) -> float:
        """Calculate string similarity."""
        if s1 == s2:
            return 1.0

        # Simple character overlap
        chars1 = set(s1.lower())
        chars2 = set(s2.lower())

        if not chars1 or not chars2:
            return 0.0

        return len(chars1 & chars2) / len(chars1 | chars2)

    def _index_episode(self, episode: Episode):
        """Index episode for retrieval."""
        # Index by context keys
        for key in episode.context.keys():
            self.episode_index[key].add(episode.id)

        # Index by tags
        for tag in episode.tags:
            self.tag_index[tag].add(episode.id)

        # Temporal index
        import bisect

        bisect.insort(self.temporal_index, (episode.start_time, episode.id))

    def _update_indices(self, episode: Episode):
        """Update all indices for episode."""
        # Embedding index
        if episode.embedding is not None:
            self.embedding_index[episode.id] = episode.embedding

        # Value index
        import bisect

        bisect.insort(self.value_index, (episode.value, episode.id))

    def _calculate_episode_importance(self, episode: Episode) -> float:
        """Calculate episode importance."""
        factors = []

        # Duration factor
        duration = episode.compute_duration()
        duration_score = min(1.0, duration / 3600)  # Normalize to 1 hour
        factors.append(duration_score * 0.2)

        # Event count factor
        event_score = min(1.0, len(episode.events) / 50)
        factors.append(event_score * 0.2)

        # Value factor
        value_score = abs(episode.value)
        factors.append(value_score * 0.3)

        # Emotional factor
        emotion_score = abs(episode.emotional_valence)
        factors.append(emotion_score * 0.2)

        # Context richness
        context_score = min(1.0, len(episode.context) / 10)
        factors.append(context_score * 0.1)

        return np.clip(sum(factors), 0, 1)

    def _generate_episode_embedding(self, episode: Episode) -> np.ndarray:
        """Generate embedding for episode."""
        # Combine various features
        features = []

        # Context features
        context_str = json.dumps(episode.context, sort_keys=True)
        context_hash = hashlib.sha256(context_str.encode()).digest()
        features.extend(list(context_hash[:32]))

        # Event features
        if episode.events:
            event_str = json.dumps(episode.events[0], sort_keys=True, default=str)
            event_hash = hashlib.sha256(event_str.encode()).digest()
            features.extend(list(event_hash[:32]))
        else:
            features.extend([0] * 32)

        # Temporal features
        features.append(episode.start_time % 86400 / 86400)  # Time of day
        features.append(episode.compute_duration() / 3600)  # Duration in hours

        # Value features
        features.append(episode.value)
        features.append(episode.emotional_valence)

        # Convert to numpy array and normalize
        embedding = np.array(features, dtype=np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def _group_similar_episodes(self, episodes: List[Episode]) -> List[List[Episode]]:
        """Group similar episodes for consolidation."""
        if not episodes:
            return []

        groups = []
        used = set()

        for ep1 in episodes:
            if ep1.id in used:
                continue

            group = [ep1]
            used.add(ep1.id)

            for ep2 in episodes:
                if ep2.id not in used:
                    similarity = self._compute_context_similarity(
                        ep1.context, ep2.context
                    )
                    if similarity > 0.7:
                        group.append(ep2)
                        used.add(ep2.id)

            groups.append(group)

        return groups

    def _merge_episodes(self, episodes: List[Episode]) -> Episode:
        """Merge multiple episodes into one."""
        if len(episodes) == 1:
            return episodes[0]

        # Create merged episode
        merged = Episode(
            id=f"merged_{time.time()}_{id(episodes)}",
            start_time=min(ep.start_time for ep in episodes),
            end_time=max(ep.end_time or ep.start_time for ep in episodes),
            events=[],
            context={},
            tags=set(),
        )

        # Merge events
        for ep in episodes:
            merged.events.extend(ep.events)
        merged.events.sort(key=lambda e: e.get("timestamp", 0))

        # Merge contexts
        for ep in episodes:
            merged.context.update(ep.context)

        # Merge tags
        for ep in episodes:
            merged.tags.update(ep.tags)

        # Calculate aggregate values
        merged.value = np.mean([ep.value for ep in episodes])
        merged.emotional_valence = np.mean([ep.emotional_valence for ep in episodes])
        merged.importance = max(ep.importance for ep in episodes)

        # Generate new embedding
        merged.embedding = self._generate_episode_embedding(merged)

        return merged

    def _detect_episode_patterns(self, episode: Episode):
        """Detect patterns in episode sequences."""
        # Create pattern signature
        pattern_sig = self._create_pattern_signature(episode)
        pattern_hash = hashlib.sha256(pattern_sig.encode()).hexdigest()

        # Add to pattern chains
        self.episode_chains[pattern_hash].append(episode.id)

    def _create_pattern_signature(self, episode: Episode) -> str:
        """Create signature for pattern matching."""
        # Use context keys and event types
        context_keys = sorted(episode.context.keys())
        event_types = [e.get("type", "unknown") for e in episode.events[:5]]

        return f"{','.join(context_keys)}_{','.join(event_types)}"


# ============================================================
# SEMANTIC MEMORY WITH TOOL PERFORMANCE
# ============================================================


@dataclass
class Concept:
    """Concept in semantic memory."""

    id: str
    name: str
    definition: str
    attributes: Dict[str, Any]
    relationships: Dict[str, List[str]]  # relation_type -> concept_ids
    embedding: Optional[np.ndarray] = None
    confidence: float = 0.5
    frequency: int = 1
    last_accessed: float = field(default_factory=time.time)
    sources: List[str] = field(default_factory=list)


@dataclass
class ToolPerformanceConcept(Concept):
    """Extended concept for tool performance knowledge."""

    tool_type: str = "reasoning"  # probabilistic, symbolic, causal, etc.
    performance_metrics: Dict[str, float] = field(
        default_factory=dict
    )  # metric -> value
    problem_domains: List[str] = field(default_factory=list)
    success_contexts: List[Dict[str, Any]] = field(default_factory=list)
    failure_contexts: List[Dict[str, Any]] = field(default_factory=list)
    optimal_conditions: Dict[str, Any] = field(default_factory=dict)
    contraindications: List[str] = field(default_factory=list)


class SemanticMemory(BaseMemorySystem):
    """Semantic memory for facts and concepts, including tool performance."""

    def __init__(self, config: MemoryConfig):
        super().__init__(config)
        self.concepts: Dict[str, Concept] = {}
        self.relations: Dict[str, List[Tuple[str, str]]] = {}
        self.ontology = {}

        # Indices
        self.name_index: Dict[str, str] = {}  # name -> concept_id
        self.attribute_index = defaultdict(set)  # attribute -> concept_ids

        # Tool-specific indices
        self.tool_index: Dict[str, Set[str]] = defaultdict(
            set
        )  # tool_name -> concept_ids
        self.performance_index = defaultdict(
            list
        )  # metric_name -> [(concept_id, value)]
        self.domain_index = defaultdict(set)  # domain -> concept_ids

        # Knowledge graph
        self.knowledge_graph = None
        if NETWORKX_AVAILABLE:
            self.knowledge_graph = nx.DiGraph()

        # Inference cache
        self.inference_cache = {}

        # Hierarchical memory for storage
        self.hierarchical_memory = HierarchicalMemory(config)

        # Initialize tool performance concepts
        self._initialize_tool_concepts()

    def _initialize_tool_concepts(self):
        """Initialize base concepts for reasoning tools."""

        # Probabilistic reasoning tool
        self.add_tool_performance_concept(
            name="probabilistic_reasoning",
            definition="Statistical and probabilistic inference tool using belief networks and uncertainty quantification",
            tool_type="probabilistic",
            attributes={
                "strengths": ["uncertainty handling", "noisy data", "belief updating"],
                "weaknesses": ["computational complexity", "requires priors"],
                "complexity": "O(n^2) for exact inference",
            },
            problem_domains=["prediction", "classification", "uncertainty_estimation"],
        )

        # Symbolic reasoning tool
        self.add_tool_performance_concept(
            name="symbolic_reasoning",
            definition="Logic-based reasoning using formal systems and rule-based inference",
            tool_type="symbolic",
            attributes={
                "strengths": [
                    "explainability",
                    "logical consistency",
                    "proof generation",
                ],
                "weaknesses": ["brittleness", "knowledge engineering"],
                "complexity": "O(2^n) worst case",
            },
            problem_domains=["theorem_proving", "planning", "constraint_satisfaction"],
        )

        # Causal reasoning tool
        self.add_tool_performance_concept(
            name="causal_reasoning",
            definition="Causal inference and counterfactual reasoning using structural causal models",
            tool_type="causal",
            attributes={
                "strengths": [
                    "intervention analysis",
                    "counterfactuals",
                    "confounding control",
                ],
                "weaknesses": ["causal graph required", "identifiability assumptions"],
                "complexity": "O(n^3) for structure learning",
            },
            problem_domains=[
                "effect_estimation",
                "policy_evaluation",
                "root_cause_analysis",
            ],
        )

        # Temporal reasoning tool
        self.add_tool_performance_concept(
            name="temporal_reasoning",
            definition="Reasoning about time, sequences, and temporal relationships",
            tool_type="temporal",
            attributes={
                "strengths": [
                    "sequence modeling",
                    "time-series analysis",
                    "event ordering",
                ],
                "weaknesses": ["memory requirements", "long-range dependencies"],
                "complexity": "O(n*T) where T is sequence length",
            },
            problem_domains=["forecasting", "scheduling", "process_monitoring"],
        )

        # Spatial reasoning tool
        self.add_tool_performance_concept(
            name="spatial_reasoning",
            definition="Geometric and topological reasoning about spatial relationships",
            tool_type="spatial",
            attributes={
                "strengths": [
                    "visualization",
                    "path planning",
                    "constraint propagation",
                ],
                "weaknesses": [
                    "dimensionality curse",
                    "computational geometry complexity",
                ],
                "complexity": "O(n log n) for many algorithms",
            },
            problem_domains=["navigation", "design", "spatial_analysis"],
        )

    def add_tool_performance_concept(
        self,
        name: str,
        definition: str,
        tool_type: str,
        attributes: Dict[str, Any] = None,
        problem_domains: List[str] = None,
        performance_metrics: Dict[str, float] = None,
        confidence: float = 0.5,
    ) -> str:
        """Add tool performance concept to semantic memory."""

        concept_id = f"tool_concept_{hashlib.sha256(name.encode()).hexdigest()[:16]}"

        # Check if concept exists
        if name in self.name_index:
            existing_id = self.name_index[name]
            existing = self.concepts[existing_id]

            # Update if it's a tool concept
            if isinstance(existing, ToolPerformanceConcept):
                existing.frequency += 1
                existing.last_accessed = time.time()
                existing.confidence = min(1.0, existing.confidence + 0.05)

                # Update metrics
                if performance_metrics:
                    for metric, value in performance_metrics.items():
                        # Exponential moving average
                        alpha = 0.3
                        if metric in existing.performance_metrics:
                            existing.performance_metrics[metric] = (
                                alpha * value
                                + (1 - alpha) * existing.performance_metrics[metric]
                            )
                        else:
                            existing.performance_metrics[metric] = value

                return existing_id
            else:
                # Convert to tool concept
                concept_id = existing_id

        concept = ToolPerformanceConcept(
            id=concept_id,
            name=name,
            definition=definition,
            tool_type=tool_type,
            attributes=attributes or {},
            relationships={},
            performance_metrics=performance_metrics or {},
            problem_domains=problem_domains or [],
            confidence=confidence,
        )

        # Generate embedding
        concept.embedding = self._generate_tool_concept_embedding(concept)

        # Store concept
        self.concepts[concept_id] = concept
        self.name_index[name] = concept_id

        # Index attributes
        for attr_key in concept.attributes:
            self.attribute_index[attr_key].add(concept_id)

        # Index tool-specific
        self.tool_index[name].add(concept_id)

        for domain in concept.problem_domains:
            self.domain_index[domain].add(concept_id)

        for metric, value in concept.performance_metrics.items():
            self.performance_index[metric].append((concept_id, value))

        # Add to knowledge graph
        if self.knowledge_graph is not None:
            self.knowledge_graph.add_node(concept_id, concept=concept)

        # Store in hierarchical memory
        self.hierarchical_memory.store(
            content={
                "name": name,
                "definition": definition,
                "attributes": attributes,
                "tool_type": tool_type,
                "domains": problem_domains,
            },
            memory_type=MemoryType.SEMANTIC,
            importance=confidence,
        )

        return concept_id

    def update_tool_performance(
        self,
        tool_name: str,
        problem_type: str,
        success: bool,
        metrics: Dict[str, float],
        context: Dict[str, Any] = None,
    ):
        """Update tool performance knowledge based on execution results."""

        # Find or create tool concept
        if tool_name in self.name_index:
            concept_id = self.name_index[tool_name]
            concept = self.concepts[concept_id]
        else:
            # Create new concept
            concept_id = self.add_tool_performance_concept(
                name=tool_name,
                definition=f"Tool for {problem_type} problems",
                tool_type=problem_type,
                performance_metrics=metrics,
            )
            concept = self.concepts[concept_id]

        if isinstance(concept, ToolPerformanceConcept):
            # Update metrics
            for metric, value in metrics.items():
                if metric in concept.performance_metrics:
                    # Running average
                    alpha = 0.2
                    concept.performance_metrics[metric] = (
                        alpha * value
                        + (1 - alpha) * concept.performance_metrics[metric]
                    )
                else:
                    concept.performance_metrics[metric] = value

            # Update success/failure contexts
            if context:
                if success:
                    concept.success_contexts.append(context)
                    # Keep only recent contexts
                    if len(concept.success_contexts) > 100:
                        concept.success_contexts = concept.success_contexts[-100:]
                else:
                    concept.failure_contexts.append(context)
                    if len(concept.failure_contexts) > 100:
                        concept.failure_contexts = concept.failure_contexts[-100:]

            # Update problem domains
            if problem_type not in concept.problem_domains:
                concept.problem_domains.append(problem_type)
                self.domain_index[problem_type].add(concept_id)

            # Update confidence based on success
            if success:
                concept.confidence = min(1.0, concept.confidence * 1.1)
            else:
                concept.confidence = max(0.1, concept.confidence * 0.95)

            # Learn optimal conditions
            if success and context:
                self._update_optimal_conditions(concept, context)

            # Learn contraindications
            if not success and context:
                self._update_contraindications(concept, context)

    def get_tool_recommendations(
        self,
        problem_type: str,
        context: Dict[str, Any] = None,
        min_confidence: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Get tool recommendations for a problem type."""

        recommendations = []

        # Find tools for this domain
        if problem_type in self.domain_index:
            concept_ids = self.domain_index[problem_type]

            for concept_id in concept_ids:
                concept = self.concepts[concept_id]

                if isinstance(concept, ToolPerformanceConcept):
                    if concept.confidence >= min_confidence:
                        # Calculate suitability score
                        suitability = self._calculate_tool_suitability(concept, context)

                        recommendation = {
                            "tool_name": concept.name,
                            "confidence": concept.confidence,
                            "suitability": suitability,
                            "performance_metrics": concept.performance_metrics.copy(),
                            "strengths": concept.attributes.get("strengths", []),
                            "weaknesses": concept.attributes.get("weaknesses", []),
                        }

                        # Check contraindications
                        if context and self._has_contraindications(concept, context):
                            recommendation["warning"] = (
                                "Has contraindications for this context"
                            )
                            recommendation["suitability"] *= 0.5

                        recommendations.append(recommendation)

        # Sort by suitability * confidence
        recommendations.sort(
            key=lambda x: x["suitability"] * x["confidence"], reverse=True
        )

        return recommendations

    def find_tool_relationships(self, tool_name: str) -> Dict[str, List[str]]:
        """Find relationships between tools."""

        if tool_name not in self.name_index:
            return {}

        concept_id = self.name_index[tool_name]
        concept = self.concepts[concept_id]

        relationships = {
            "complements": [],
            "substitutes": [],
            "requires": [],
            "conflicts_with": [],
        }

        # Find relationships through the knowledge graph
        if self.knowledge_graph is not None and concept_id in self.knowledge_graph:
            # Direct relationships
            for successor in self.knowledge_graph.successors(concept_id):
                edge_data = self.knowledge_graph[concept_id][successor]
                relation = edge_data.get("relation", "related")

                successor_concept = self.concepts.get(successor)
                if successor_concept:
                    if relation in relationships:
                        relationships[relation].append(successor_concept.name)

            # Inferred relationships based on problem domains
            for other_id, other_concept in self.concepts.items():
                if other_id != concept_id and isinstance(
                    other_concept, ToolPerformanceConcept
                ):
                    # Check domain overlap
                    domain_overlap = set(concept.problem_domains) & set(
                        other_concept.problem_domains
                    )

                    if domain_overlap:
                        # Similar performance suggests substitute
                        perf_similarity = self._compute_performance_similarity(
                            concept, other_concept
                        )

                        if perf_similarity > 0.8:
                            relationships["substitutes"].append(other_concept.name)
                        elif perf_similarity > 0.5:
                            relationships["complements"].append(other_concept.name)

        return relationships

    def _generate_tool_concept_embedding(
        self, concept: ToolPerformanceConcept
    ) -> np.ndarray:
        """Generate embedding for tool concept."""
        # Combine various features
        text = f"{concept.name} {concept.definition} {concept.tool_type}"

        for key, value in concept.attributes.items():
            text += f" {key}:{value}"

        for domain in concept.problem_domains:
            text += f" domain:{domain}"

        # Add performance metrics
        for metric, value in concept.performance_metrics.items():
            text += f" {metric}:{value:.2f}"

        # Hash-based embedding
        text_hash = hashlib.sha256(text.encode()).digest()
        embedding = np.frombuffer(text_hash, dtype=np.uint8).astype(np.float32)

        # Add numeric features
        extra_features = [
            concept.confidence,
            len(concept.problem_domains) / 10,
            len(concept.performance_metrics) / 10,
            (
                np.mean(list(concept.performance_metrics.values()))
                if concept.performance_metrics
                else 0
            ),
        ]

        # Combine and normalize
        combined = np.concatenate([embedding[:60], extra_features])
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm

        return combined

    def _calculate_tool_suitability(
        self, concept: ToolPerformanceConcept, context: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate tool suitability for context."""

        suitability = concept.confidence

        if not context:
            return suitability

        # Check success contexts similarity
        if concept.success_contexts:
            max_similarity = 0
            for success_ctx in concept.success_contexts[-10:]:  # Check recent successes
                similarity = self._compute_context_similarity(context, success_ctx)
                max_similarity = max(max_similarity, similarity)

            suitability *= 0.5 + 0.5 * max_similarity

        # Check failure contexts
        if concept.failure_contexts:
            max_similarity = 0
            for failure_ctx in concept.failure_contexts[-10:]:
                similarity = self._compute_context_similarity(context, failure_ctx)
                max_similarity = max(max_similarity, similarity)

            # Reduce suitability if similar to failures
            suitability *= 1.0 - 0.3 * max_similarity

        # Check optimal conditions match
        if concept.optimal_conditions:
            condition_match = self._compute_context_similarity(
                context, concept.optimal_conditions
            )
            suitability *= 0.7 + 0.3 * condition_match

        return np.clip(suitability, 0, 1)

    def _compute_context_similarity(self, ctx1: Dict, ctx2: Dict) -> float:
        """Compute similarity between contexts."""
        if not ctx1 or not ctx2:
            return 0.0

        keys1 = set(ctx1.keys())
        keys2 = set(ctx2.keys())

        if not keys1 or not keys2:
            return 0.0

        # Key overlap
        key_overlap = len(keys1 & keys2) / len(keys1 | keys2)

        # Value similarity
        value_sim = 0
        shared = keys1 & keys2

        for key in shared:
            if ctx1[key] == ctx2[key]:
                value_sim += 1
            elif isinstance(ctx1[key], (int, float)) and isinstance(
                ctx2[key], (int, float)
            ):
                diff = abs(ctx1[key] - ctx2[key])
                max_val = max(abs(ctx1[key]), abs(ctx2[key]), 1)
                value_sim += max(0, 1 - diff / max_val)

        if shared:
            value_sim /= len(shared)

        return 0.5 * key_overlap + 0.5 * value_sim

    def _compute_performance_similarity(
        self, tool1: ToolPerformanceConcept, tool2: ToolPerformanceConcept
    ) -> float:
        """Compute similarity between tool performances."""

        if not tool1.performance_metrics or not tool2.performance_metrics:
            return 0.0

        metrics1 = set(tool1.performance_metrics.keys())
        metrics2 = set(tool2.performance_metrics.keys())

        shared = metrics1 & metrics2
        if not shared:
            return 0.0

        similarity = 0
        for metric in shared:
            val1 = tool1.performance_metrics[metric]
            val2 = tool2.performance_metrics[metric]

            if val1 == val2:
                similarity += 1
            else:
                max_val = max(abs(val1), abs(val2), 1)
                similarity += max(0, 1 - abs(val1 - val2) / max_val)

        return similarity / len(shared)

    def _update_optimal_conditions(
        self, concept: ToolPerformanceConcept, context: Dict[str, Any]
    ):
        """Update optimal conditions based on successful execution."""

        if not concept.optimal_conditions:
            concept.optimal_conditions = context.copy()
        else:
            # Merge conditions
            for key, value in context.items():
                if key not in concept.optimal_conditions:
                    concept.optimal_conditions[key] = value
                elif isinstance(value, (int, float)) and isinstance(
                    concept.optimal_conditions[key], (int, float)
                ):
                    # Average numeric values
                    concept.optimal_conditions[key] = (
                        0.7 * concept.optimal_conditions[key] + 0.3 * value
                    )

    def _update_contraindications(
        self, concept: ToolPerformanceConcept, context: Dict[str, Any]
    ):
        """Update contraindications based on failed execution."""

        # Extract key patterns from failure context
        patterns = []

        for key, value in context.items():
            if isinstance(value, bool) and value:
                patterns.append(f"{key}=True")
            elif isinstance(value, (int, float)):
                if value > 0.8:
                    patterns.append(f"{key}>0.8")
                elif value < 0.2:
                    patterns.append(f"{key}<0.2")

        # Add unique patterns
        for pattern in patterns:
            if pattern not in concept.contraindications:
                concept.contraindications.append(pattern)

        # Keep only recent contraindications
        if len(concept.contraindications) > 50:
            concept.contraindications = concept.contraindications[-50:]

    def _has_contraindications(
        self, concept: ToolPerformanceConcept, context: Dict[str, Any]
    ) -> bool:
        """Check if context has contraindications for tool."""

        for contraindication in concept.contraindications:
            if "=" in contraindication:
                key, value = contraindication.split("=")
                if key in context and str(context[key]) == value:
                    return True
            elif ">" in contraindication:
                key, threshold = contraindication.split(">")
                if key in context and isinstance(context[key], (int, float)):
                    if context[key] > float(threshold):
                        return True
            elif "<" in contraindication:
                key, threshold = contraindication.split("<")
                if key in context and isinstance(context[key], (int, float)):
                    if context[key] < float(threshold):
                        return True

        return False

    # Keep all original methods unchanged

    def add_concept(
        self,
        name: str,
        definition: str,
        attributes: Dict[str, Any] = None,
        confidence: float = 0.5,
    ) -> str:
        """Add new concept."""
        concept_id = f"concept_{hashlib.sha256(name.encode()).hexdigest()[:16]}"

        # Check if concept exists
        if name in self.name_index:
            existing_id = self.name_index[name]
            existing = self.concepts[existing_id]
            existing.frequency += 1
            existing.last_accessed = time.time()

            # Update confidence
            existing.confidence = min(1.0, existing.confidence + 0.1)

            return existing_id

        concept = Concept(
            id=concept_id,
            name=name,
            definition=definition,
            attributes=attributes or {},
            relationships={},
            confidence=confidence,
        )

        # Generate embedding
        concept.embedding = self._generate_concept_embedding(concept)

        # Store concept
        self.concepts[concept_id] = concept
        self.name_index[name] = concept_id

        # Index attributes
        for attr_key in concept.attributes:
            self.attribute_index[attr_key].add(concept_id)

        # Add to knowledge graph
        if self.knowledge_graph is not None:
            self.knowledge_graph.add_node(concept_id, concept=concept)

        # Store in hierarchical memory
        self.hierarchical_memory.store(
            content={"name": name, "definition": definition, "attributes": attributes},
            memory_type=MemoryType.SEMANTIC,
            importance=confidence,
        )

        return concept_id

    def add_relation(
        self, concept1_id: str, relation: str, concept2_id: str, confidence: float = 0.5
    ):
        """Add relation between concepts."""
        if concept1_id not in self.concepts or concept2_id not in self.concepts:
            return

        # Add to concept relationships
        if relation not in self.concepts[concept1_id].relationships:
            self.concepts[concept1_id].relationships[relation] = []

        if concept2_id not in self.concepts[concept1_id].relationships[relation]:
            self.concepts[concept1_id].relationships[relation].append(concept2_id)

        # Track relation
        if relation not in self.relations:
            self.relations[relation] = []
        self.relations[relation].append((concept1_id, concept2_id))

        # Add to knowledge graph
        if self.knowledge_graph is not None:
            self.knowledge_graph.add_edge(
                concept1_id, concept2_id, relation=relation, confidence=confidence
            )

        # Clear inference cache
        self.inference_cache.clear()

    def query_concept(self, name: str) -> Optional[Concept]:
        """Query concept by name."""
        if name in self.name_index:
            concept_id = self.name_index[name]
            concept = self.concepts[concept_id]
            concept.last_accessed = time.time()
            return concept
        return None

    def store(self, content: Any, **kwargs) -> Memory:
        """Store semantic knowledge."""
        # Extract concept from content
        if isinstance(content, dict):
            name = content.get("name", str(content))
            definition = content.get("definition", str(content))
            attributes = content.get("attributes", {})
        else:
            name = str(content)[:100]
            definition = str(content)
            attributes = kwargs.get("attributes", {})

        # Add concept
        concept_id = self.add_concept(
            name, definition, attributes, confidence=kwargs.get("importance", 0.5)
        )

        concept = self.concepts[concept_id]

        # Create memory
        memory = Memory(
            id=concept_id,
            type=MemoryType.SEMANTIC,
            content=content,
            embedding=concept.embedding,
            timestamp=time.time(),
            importance=concept.confidence,
            metadata={"concept_name": name},
        )

        self.stats.total_stores += 1

        return memory

    def retrieve(self, query: MemoryQuery) -> RetrievalResult:
        """Retrieve semantic memories."""
        start_time = time.time()

        # Search concepts
        results = []

        if query.content:
            if isinstance(query.content, str):
                # Search by name
                concept = self.query_concept(query.content)
                if concept:
                    results.append((concept, 1.0))

                # Search by partial match
                for name, concept_id in self.name_index.items():
                    if query.content.lower() in name.lower():
                        results.append((self.concepts[concept_id], 0.5))

            elif isinstance(query.content, dict):
                # Search by attributes
                query_attrs = query.content.get("attributes", {})
                for concept in self.concepts.values():
                    score = self._compute_attribute_similarity(
                        query_attrs, concept.attributes
                    )
                    if score > 0.3:
                        results.append((concept, score))

        # Search by embedding
        if query.embedding is not None:
            for concept in self.concepts.values():
                if concept.embedding is not None:
                    similarity = np.dot(query.embedding, concept.embedding)
                    if similarity > 0.5:
                        results.append((concept, float(similarity)))

        # Sort and convert to memories
        results.sort(key=lambda x: x[1], reverse=True)
        results = results[: query.limit]

        memories = []
        scores = []

        for concept, score in results:
            memory = Memory(
                id=concept.id,
                type=MemoryType.SEMANTIC,
                content={
                    "name": concept.name,
                    "definition": concept.definition,
                    "attributes": concept.attributes,
                },
                embedding=concept.embedding,
                timestamp=concept.last_accessed,
                importance=concept.confidence,
            )
            memories.append(memory)
            scores.append(score)

        result = RetrievalResult(
            memories=memories,
            scores=scores,
            query_time_ms=(time.time() - start_time) * 1000,
            total_matches=len(results),
        )

        self.stats.total_queries += 1

        return result

    def forget(self, memory_id: str) -> bool:
        """Remove semantic memory."""
        if memory_id in self.concepts:
            concept = self.concepts[memory_id]

            # Remove from indices
            if concept.name in self.name_index:
                del self.name_index[concept.name]

            for attr in concept.attributes:
                self.attribute_index[attr].discard(memory_id)

            # Remove from tool indices if it's a tool concept
            if isinstance(concept, ToolPerformanceConcept):
                self.tool_index[concept.name].discard(memory_id)

                for domain in concept.problem_domains:
                    self.domain_index[domain].discard(memory_id)

            # Remove from graph
            if self.knowledge_graph is not None and memory_id in self.knowledge_graph:
                self.knowledge_graph.remove_node(memory_id)

            # Remove from concepts
            del self.concepts[memory_id]

            # Clear cache
            self.inference_cache.clear()

            self.stats.total_memories -= 1
            return True

        return False

    def consolidate(self) -> int:
        """Consolidate semantic memories."""
        # Merge similar concepts
        similar_groups = self._find_similar_concepts()
        consolidated = 0

        for group in similar_groups:
            if len(group) > 1:
                merged = self._merge_concepts(group)

                # Remove old concepts
                for concept in group:
                    if concept.id != merged.id:
                        self.forget(concept.id)
                        consolidated += 1

        # Prune weak relations
        if self.knowledge_graph is not None:
            weak_edges = [
                (u, v)
                for u, v, data in self.knowledge_graph.edges(data=True)
                if data.get("confidence", 1.0) < 0.3
            ]
            self.knowledge_graph.remove_edges_from(weak_edges)
            consolidated += len(weak_edges)

        self.stats.total_consolidations += 1

        return consolidated

    def infer_relations(
        self, concept_id: str, max_depth: int = 2
    ) -> Dict[str, List[str]]:
        """Infer implicit relations."""
        # Check cache
        cache_key = f"{concept_id}_{max_depth}"
        if cache_key in self.inference_cache:
            return self.inference_cache[cache_key]

        inferred = {}

        if concept_id not in self.concepts:
            return inferred

        if self.knowledge_graph is None:
            # Simple inference without graph
            inferred = self._infer_relations_simple(concept_id, max_depth)
        else:
            # Graph-based inference
            inferred = self._infer_relations_graph(concept_id, max_depth)

        # Cache result
        self.inference_cache[cache_key] = inferred

        return inferred

    def _infer_relations_simple(
        self, concept_id: str, max_depth: int
    ) -> Dict[str, List[str]]:
        """Simple transitive relation inference."""
        inferred = {}
        visited = set()

        def explore(cid: str, depth: int, path: List[str]):
            if depth >= max_depth or cid in visited:
                return

            visited.add(cid)

            if cid in self.concepts:
                concept = self.concepts[cid]

                for rel_type, related_ids in concept.relationships.items():
                    for related_id in related_ids:
                        # Direct relation
                        if depth == 0:
                            if rel_type not in inferred:
                                inferred[rel_type] = []
                            if related_id not in inferred[rel_type]:
                                inferred[rel_type].append(related_id)
                        else:
                            # Transitive relation
                            full_path = path + [rel_type]
                            path_key = "->".join(full_path)
                            if path_key not in inferred:
                                inferred[path_key] = []
                            if related_id not in inferred[path_key]:
                                inferred[path_key].append(related_id)

                        # Recurse
                        explore(related_id, depth + 1, path + [rel_type])

        explore(concept_id, 0, [])

        return inferred

    def _infer_relations_graph(
        self, concept_id: str, max_depth: int
    ) -> Dict[str, List[str]]:
        """Graph-based relation inference."""
        inferred = {}

        # Find all paths from concept
        for target in self.knowledge_graph.nodes():
            if target != concept_id:
                try:
                    paths = list(
                        nx.all_simple_paths(
                            self.knowledge_graph, concept_id, target, cutoff=max_depth
                        )
                    )

                    for path in paths:
                        if len(path) > 1:
                            # Get relation chain
                            relations = []
                            for i in range(len(path) - 1):
                                edge_data = self.knowledge_graph[path[i]][path[i + 1]]
                                relations.append(edge_data.get("relation", "related"))

                            relation_key = "->".join(relations)
                            if relation_key not in inferred:
                                inferred[relation_key] = []
                            inferred[relation_key].append(target)

                except nx.NetworkXNoPath:
                    logger.debug(f"Operation failed: {e}")

        return inferred

    def _generate_concept_embedding(self, concept: Concept) -> np.ndarray:
        """Generate embedding for concept."""
        # Combine name, definition, and attributes
        text = f"{concept.name} {concept.definition}"
        for key, value in concept.attributes.items():
            text += f" {key}:{value}"

        # Simple hash-based embedding
        text_hash = hashlib.sha256(text.encode()).digest()
        embedding = np.frombuffer(text_hash, dtype=np.uint8).astype(np.float32)

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def _compute_attribute_similarity(self, attrs1: Dict, attrs2: Dict) -> float:
        """Compute similarity between attribute sets."""
        if not attrs1 or not attrs2:
            return 0.0

        keys1 = set(attrs1.keys())
        keys2 = set(attrs2.keys())

        # Key overlap
        key_overlap = len(keys1 & keys2) / len(keys1 | keys2) if keys1 | keys2 else 0

        # Value similarity for shared keys
        value_sim = 0
        shared = keys1 & keys2

        for key in shared:
            if attrs1[key] == attrs2[key]:
                value_sim += 1

        if shared:
            value_sim /= len(shared)

        return 0.6 * key_overlap + 0.4 * value_sim

    def _find_similar_concepts(self, threshold: float = 0.8) -> List[List[Concept]]:
        """Find groups of similar concepts."""
        groups = []
        used = set()

        concepts = list(self.concepts.values())

        for i, c1 in enumerate(concepts):
            if c1.id in used:
                continue

            group = [c1]
            used.add(c1.id)

            for c2 in concepts[i + 1 :]:
                if c2.id not in used:
                    # Check name similarity
                    if c1.name.lower() == c2.name.lower():
                        group.append(c2)
                        used.add(c2.id)
                    # Check embedding similarity (only if both have embeddings and same shape)
                    elif (
                        c1.embedding is not None
                        and c2.embedding is not None
                        and c1.embedding.shape == c2.embedding.shape
                    ):
                        sim = np.dot(c1.embedding, c2.embedding)
                        if sim > threshold:
                            group.append(c2)
                            used.add(c2.id)

            if len(group) > 1:
                groups.append(group)

        return groups

    def _merge_concepts(self, concepts: List[Concept]) -> Concept:
        """Merge multiple concepts."""
        if len(concepts) == 1:
            return concepts[0]

        # Use most confident as base
        base = max(concepts, key=lambda c: c.confidence)

        # Handle tool concepts specially
        if isinstance(base, ToolPerformanceConcept):
            merged = copy.deepcopy(base)

            for concept in concepts:
                if isinstance(concept, ToolPerformanceConcept):
                    # Merge performance metrics
                    for metric, value in concept.performance_metrics.items():
                        if metric in merged.performance_metrics:
                            # Average
                            merged.performance_metrics[metric] = (
                                merged.performance_metrics[metric] + value
                            ) / 2
                        else:
                            merged.performance_metrics[metric] = value

                    # Merge domains
                    merged.problem_domains.extend(concept.problem_domains)
                    merged.problem_domains = list(set(merged.problem_domains))

                    # Merge contexts
                    merged.success_contexts.extend(concept.success_contexts)
                    merged.failure_contexts.extend(concept.failure_contexts)
        else:
            merged = Concept(
                id=base.id,
                name=base.name,
                definition=base.definition,
                attributes=base.attributes.copy(),
                relationships=base.relationships.copy(),
                embedding=base.embedding,
                confidence=max(c.confidence for c in concepts),
                frequency=sum(c.frequency for c in concepts),
                last_accessed=max(c.last_accessed for c in concepts),
                sources=[],
            )

        # Merge attributes
        for concept in concepts:
            merged.attributes.update(concept.attributes)
            merged.sources.extend(concept.sources)

        # Merge relationships
        for concept in concepts:
            for rel_type, targets in concept.relationships.items():
                if rel_type not in merged.relationships:
                    merged.relationships[rel_type] = []
                merged.relationships[rel_type].extend(targets)

        # Deduplicate
        for rel_type in merged.relationships:
            merged.relationships[rel_type] = set(merged.relationships[rel_type])

        merged.sources = list(set(merged.sources))

        return merged


# ============================================================
# PROCEDURAL MEMORY
# ============================================================


@dataclass
class Skill:
    """Skill in procedural memory."""

    id: str
    name: str
    steps: List[Union[Callable, str]]  # Can be functions or descriptions
    preconditions: List[str]
    postconditions: List[str]
    performance_history: List[float]
    success_rate: float = 0.0
    execution_time: float = 0.0
    dependencies: List[str] = field(default_factory=list)

    def execute(self, context: Dict[str, Any]) -> Any:
        """Execute skill."""
        start_time = time.time()

        try:
            # Check preconditions
            for condition in self.preconditions:
                if not self._check_condition(condition, context):
                    raise Exception(f"Precondition failed: {condition}")

            # Execute steps
            result = context
            for i, step in enumerate(self.steps):
                try:
                    if callable(step):
                        result = step(result)
                    else:
                        result = self._execute_string_step(step, result)
                except Exception as e:
                    logger.error(f"Step {i} failed: {e}")
                    raise

            # Verify postconditions
            for condition in self.postconditions:
                if not self._check_condition(condition, result):
                    logger.warning(f"Postcondition failed: {condition}")

            # Update execution time
            self.execution_time = time.time() - start_time

            return result

        except Exception as e:
            logger.error(f"Skill execution failed: {e}")
            self.execution_time = time.time() - start_time
            raise

    def _check_condition(self, condition: str, context: Any) -> bool:
        """Check if condition is met using safe expression evaluation.

        Fails securely by returning False when unsafe conditions are detected.
        
        FIX #MEM-7: Context is now sanitized to only allow primitive types,
        preventing code injection through malicious context values.
        """
        try:
            # Parse condition as Python expression using AST
            if isinstance(context, dict):
                # Use ast module for safe evaluation of simple expressions
                # This only allows literal structures and basic comparisons

                try:
                    import ast

                    # Parse the condition
                    tree = ast.parse(condition, mode="eval")

                    # Only allow safe operations (Python 3.8+ compatible)
                    # Removed deprecated ast.Num, ast.Str, ast.NameConstant
                    safe_nodes = (
                        ast.Expression,
                        ast.Constant,
                        ast.List,
                        ast.Tuple,
                        ast.Dict,
                        ast.Name,
                        ast.Load,
                        ast.Compare,
                        ast.BoolOp,
                        ast.And,
                        ast.Or,
                        ast.Eq,
                        ast.NotEq,
                        ast.Lt,
                        ast.LtE,
                        ast.Gt,
                        ast.GtE,
                        ast.In,
                        ast.NotIn,
                        ast.UnaryOp,
                        ast.Not,
                    )

                    for node in ast.walk(tree):
                        if not isinstance(node, safe_nodes):
                            logger.warning(
                                f"Unsafe node in condition '{condition}': {type(node).__name__}"
                            )
                            # Fail securely - return False instead of True
                            return False

                    # FIX #MEM-7: Sanitize context - only allow primitive types
                    # This prevents code injection through malicious context values
                    # like {"__import__": __import__, "os": os}
                    safe_context = {}
                    for key, value in context.items():
                        # Reject keys that could be used for injection
                        if key.startswith("_"):
                            logger.warning(
                                f"Skipping underscore-prefixed context key: {key}"
                            )
                            continue
                        
                        # Only allow primitive types
                        if isinstance(value, (str, int, float, bool, type(None))):
                            safe_context[key] = value
                        elif isinstance(value, (list, tuple)):
                            # Allow lists/tuples of primitives only
                            if all(isinstance(v, (str, int, float, bool, type(None))) for v in value):
                                safe_context[key] = value
                            else:
                                logger.debug(
                                    f"Skipping non-primitive list/tuple context var: {key}"
                                )
                        elif isinstance(value, dict):
                            # Allow dicts with string keys and primitive values
                            if all(
                                isinstance(k, str) and isinstance(v, (str, int, float, bool, type(None)))
                                for k, v in value.items()
                            ):
                                safe_context[key] = value
                            else:
                                logger.debug(
                                    f"Skipping complex dict context var: {key}"
                                )
                        else:
                            logger.debug(
                                f"Skipping non-primitive context var: {key} (type={type(value).__name__})"
                            )

                    # Create a restricted namespace with only safe context variables
                    # and no builtins to prevent code injection
                    namespace = {"__builtins__": {}}
                    namespace.update(safe_context)

                    # Compile and evaluate the safe expression
                    # nosec B307: Using eval with restricted namespace (no builtins) for safe evaluation
                    code = compile(tree, "<condition>", "eval")
                    return bool(eval(code, namespace, {}))  # nosec B307

                except (SyntaxError, ValueError, TypeError) as e:
                    logger.debug(f"Could not parse condition '{condition}': {e}")
                    # Fail securely - return False if we can't parse
                    return False
            else:
                # Simple existence check
                return context is not None
        except Exception as e:
            logger.warning(f"Condition check failed for '{condition}': {e}")
            # Fail securely - return False on any unexpected error
            return False

    def _execute_string_step(self, step: str, context: Any) -> Any:
        """Execute string-based step."""
        # Simple implementation - could be enhanced
        logger.info(f"Executing step: {step}")
        return context

    def update_performance(self, success: bool, execution_time: float):
        """Update performance metrics."""
        self.performance_history.append(1.0 if success else 0.0)

        # Update success rate (exponential moving average)
        alpha = 0.1
        self.success_rate = (
            alpha * (1.0 if success else 0.0) + (1 - alpha) * self.success_rate
        )

        # Update execution time
        self.execution_time = alpha * execution_time + (1 - alpha) * self.execution_time


class ProceduralMemory(BaseMemorySystem):
    """Procedural memory for skills and procedures."""

    def __init__(self, config: MemoryConfig):
        super().__init__(config)
        self.skills: Dict[str, Skill] = {}
        self.skill_hierarchy = {}
        self.skill_chains = defaultdict(list)  # skill_id -> [next_skill_ids]

        # Performance tracking
        self.execution_history = []

        # Skill graph
        self.skill_graph = None
        if NETWORKX_AVAILABLE:
            self.skill_graph = nx.DiGraph()

        # Hierarchical memory
        self.hierarchical_memory = HierarchicalMemory(config)

    def add_skill(
        self,
        name: str,
        steps: List[Union[Callable, str]],
        preconditions: List[str] = None,
        postconditions: List[str] = None,
        dependencies: List[str] = None,
    ) -> str:
        """Add new skill."""
        skill_id = f"skill_{hashlib.sha256(name.encode()).hexdigest()[:16]}"

        # Check if skill exists
        existing = self.find_skill(name)
        if existing:
            return existing.id

        skill = Skill(
            id=skill_id,
            name=name,
            steps=steps,
            preconditions=preconditions or [],
            postconditions=postconditions or [],
            performance_history=[],
            dependencies=dependencies or [],
        )

        self.skills[skill_id] = skill

        # Add to skill graph
        if self.skill_graph is not None:
            self.skill_graph.add_node(skill_id, skill=skill)

            # Add dependency edges
            for dep in skill.dependencies:
                dep_skill = self.find_skill(dep)
                if dep_skill:
                    self.skill_graph.add_edge(
                        dep_skill.id, skill_id, relation="required_for"
                    )

        # Store in hierarchical memory
        self.hierarchical_memory.store(
            content={"name": name, "steps": str(steps)},
            memory_type=MemoryType.PROCEDURAL,
            importance=0.5,
        )

        return skill_id

    def execute_skill(self, skill_name: str, context: Dict[str, Any]) -> Any:
        """Execute skill by name."""
        skill = self.find_skill(skill_name)

        if not skill:
            raise ValueError(f"Skill not found: {skill_name}")

        # Check dependencies
        for dep_name in skill.dependencies:
            dep_skill = self.find_skill(dep_name)
            if dep_skill and dep_skill.success_rate < 0.5:
                logger.warning(f"Dependency {dep_name} has low success rate")

        start_time = time.time()
        success = False
        result = None

        try:
            result = skill.execute(context)
            success = True
        except Exception as e:
            logger.error(f"Skill execution failed: {e}")
            result = None

        # Update performance
        execution_time = time.time() - start_time
        skill.update_performance(success, execution_time)

        # Record execution
        self.execution_history.append(
            {
                "skill_id": skill.id,
                "timestamp": time.time(),
                "success": success,
                "execution_time": execution_time,
            }
        )

        return result

    def store(self, content: Any, **kwargs) -> Memory:
        """Store procedural memory."""
        # Extract skill information
        if isinstance(content, dict):
            name = content.get("name", "unnamed")
            steps = content.get("steps", [])
            preconditions = content.get("preconditions", [])
            postconditions = content.get("postconditions", [])
        else:
            name = str(content)[:50]
            steps = [str(content)]
            preconditions = []
            postconditions = []

        # Add skill
        skill_id = self.add_skill(name, steps, preconditions, postconditions)
        self.skills[skill_id]

        # Create memory
        memory = Memory(
            id=skill_id,
            type=MemoryType.PROCEDURAL,
            content=content,
            timestamp=time.time(),
            importance=kwargs.get("importance", 0.5),
            metadata={"skill_name": name},
        )

        self.stats.total_stores += 1

        return memory

    def retrieve(self, query: MemoryQuery) -> RetrievalResult:
        """Retrieve procedural memories."""
        start_time = time.time()

        results = []

        # Search by name
        if query.content:
            if isinstance(query.content, str):
                skill = self.find_skill(query.content)
                if skill:
                    results.append((skill, 1.0))

                # Partial match
                for skill in self.skills.values():
                    if query.content.lower() in skill.name.lower():
                        results.append((skill, 0.5))

        # Filter by success rate
        if "min_success_rate" in query.filters:
            min_rate = query.filters["min_success_rate"]
            results = [(s, score) for s, score in results if s.success_rate >= min_rate]

        # Sort and convert
        results.sort(key=lambda x: x[1], reverse=True)
        results = results[: query.limit]

        memories = []
        scores = []

        for skill, score in results:
            memory = Memory(
                id=skill.id,
                type=MemoryType.PROCEDURAL,
                content={"name": skill.name, "steps": skill.steps},
                timestamp=time.time(),
                importance=skill.success_rate,
                metadata={
                    "execution_time": skill.execution_time,
                    "success_rate": skill.success_rate,
                },
            )
            memories.append(memory)
            scores.append(score)

        result = RetrievalResult(
            memories=memories,
            scores=scores,
            query_time_ms=(time.time() - start_time) * 1000,
            total_matches=len(results),
        )

        self.stats.total_queries += 1

        return result

    def forget(self, memory_id: str) -> bool:
        """Remove procedural memory."""
        if memory_id in self.skills:
            # Remove from graph
            if self.skill_graph is not None and memory_id in self.skill_graph:
                self.skill_graph.remove_node(memory_id)

            # Remove from skills
            del self.skills[memory_id]

            self.stats.total_memories -= 1
            return True

        return False

    def consolidate(self) -> int:
        """Consolidate procedural memories."""
        # Remove low-performing skills
        consolidated = 0
        threshold = 0.2

        skills_to_remove = [
            skill_id
            for skill_id, skill in self.skills.items()
            if len(skill.performance_history) > 10 and skill.success_rate < threshold
        ]

        for skill_id in skills_to_remove:
            self.forget(skill_id)
            consolidated += 1

        # Merge similar skills
        similar_groups = self._find_similar_skills()

        for group in similar_groups:
            if len(group) > 1:
                merged = self._merge_skills(group)

                for skill in group:
                    if skill.id != merged.id:
                        self.forget(skill.id)
                        consolidated += 1

        self.stats.total_consolidations += 1

        return consolidated

    def find_skill(self, name: str) -> Optional[Skill]:
        """Find skill by name."""
        for skill in self.skills.values():
            if skill.name == name:
                return skill
        return None

    def compose_skills(self, skill_names: List[str], new_name: str) -> str:
        """Compose multiple skills into new skill."""
        skills = [self.find_skill(name) for name in skill_names]
        skills = [s for s in skills if s is not None]

        if not skills:
            raise ValueError("No valid skills found")

        # Combine steps
        combined_steps = []
        combined_preconditions = []
        combined_postconditions = []
        combined_dependencies = []

        for i, skill in enumerate(skills):
            # Add preconditions from first skill
            if i == 0:
                combined_preconditions.extend(skill.preconditions)

            # Add steps
            combined_steps.extend(skill.steps)

            # Add postconditions from last skill
            if i == len(skills) - 1:
                combined_postconditions.extend(skill.postconditions)

            # Collect dependencies
            combined_dependencies.extend(skill.dependencies)

        # Create new skill
        return self.add_skill(
            new_name,
            combined_steps,
            list(set(combined_preconditions)),
            list(set(combined_postconditions)),
            list(set(combined_dependencies)),
        )

    def _find_similar_skills(self, threshold: float = 0.7) -> List[List[Skill]]:
        """Find groups of similar skills."""
        groups = []
        used = set()

        skills = list(self.skills.values())

        for i, s1 in enumerate(skills):
            if s1.id in used:
                continue

            group = [s1]
            used.add(s1.id)

            for s2 in skills[i + 1 :]:
                if s2.id not in used:
                    similarity = self._compute_skill_similarity(s1, s2)
                    if similarity > threshold:
                        group.append(s2)
                        used.add(s2.id)

            if len(group) > 1:
                groups.append(group)

        return groups

    def _compute_skill_similarity(self, s1: Skill, s2: Skill) -> float:
        """Compute similarity between skills."""
        # Name similarity
        name_sim = 1.0 if s1.name == s2.name else 0.0

        # Step overlap
        steps1 = set(str(s) for s in s1.steps)
        steps2 = set(str(s) for s in s2.steps)

        if steps1 or steps2:
            step_sim = len(steps1 & steps2) / len(steps1 | steps2)
        else:
            step_sim = 0.0

        # Condition overlap
        pre1 = set(s1.preconditions)
        pre2 = set(s2.preconditions)
        post1 = set(s1.postconditions)
        post2 = set(s2.postconditions)

        condition_sim = 0.0
        if pre1 | pre2:
            condition_sim += len(pre1 & pre2) / len(pre1 | pre2) * 0.5
        if post1 | post2:
            condition_sim += len(post1 & post2) / len(post1 | post2) * 0.5

        return 0.4 * name_sim + 0.4 * step_sim + 0.2 * condition_sim

    def _merge_skills(self, skills: List[Skill]) -> Skill:
        """Merge multiple skills."""
        if len(skills) == 1:
            return skills[0]

        # Use best performing as base
        base = max(skills, key=lambda s: s.success_rate)

        merged = copy.deepcopy(base)
        merged.performance_history = []

        # Merge performance history
        for skill in skills:
            merged.performance_history.extend(skill.performance_history)

        # Recalculate success rate
        if merged.performance_history:
            merged.success_rate = sum(merged.performance_history) / len(
                merged.performance_history
            )

        return merged


# ============================================================
# WORKING MEMORY
# ============================================================


@dataclass
class WorkingMemoryBuffer:
    """Buffer in working memory."""

    content: Any
    timestamp: float
    attention_weight: float = 1.0
    relevance: float = 1.0
    activation_level: float = 1.0
    source: str = "external"


class WorkingMemory(BaseMemorySystem):
    """Working memory with limited capacity and attention."""

    def __init__(self, config: MemoryConfig):
        super().__init__(config)
        self.capacity = config.max_working_memory
        self.buffer: deque = deque(maxlen=self.capacity)
        self.focus: Optional[WorkingMemoryBuffer] = None
        self.attention_weights = {}

        # Phonological loop and visuospatial sketchpad
        self.phonological_loop: deque = deque(maxlen=7)  # ~2 seconds of speech
        self.visuospatial_sketchpad: deque = deque(maxlen=4)  # Visual/spatial info

        # Central executive
        self.task_queue: deque = deque()
        self.current_task = None

        # Set running flag BEFORE starting thread
        self._running = True

        # Maintenance rehearsal
        self.rehearsal_thread = threading.Thread(
            target=self._rehearsal_loop, daemon=True
        )
        self.rehearsal_thread.start()

    def store(self, content: Any, **kwargs) -> Memory:
        """Store in working memory."""
        relevance = kwargs.get("relevance", 1.0)
        source = kwargs.get("source", "external")

        # Add to buffer
        item = WorkingMemoryBuffer(
            content=content, timestamp=time.time(), relevance=relevance, source=source
        )

        if len(self.buffer) >= self.capacity:
            # Remove least relevant
            min_item = min(self.buffer, key=lambda x: x.activation_level * x.relevance)
            self.buffer.remove(min_item)

        self.buffer.append(item)

        # Update focus
        if relevance > 0.8:
            self.focus = item

        # Create memory
        memory = Memory(
            id=f"wm_{time.time()}_{id(content)}",
            type=MemoryType.WORKING,
            content=content,
            timestamp=time.time(),
            importance=relevance,
            metadata={"source": source},
        )

        self.stats.total_stores += 1

        return memory

    def retrieve(self, query: MemoryQuery) -> RetrievalResult:
        """Retrieve from working memory."""
        start_time = time.time()

        memories = []
        scores = []

        for item in self.buffer:
            # Simple content matching
            score = 0.0

            if query.content:
                if item.content == query.content:
                    score = 1.0
                elif isinstance(item.content, str) and isinstance(query.content, str):
                    if query.content in item.content:
                        score = 0.5

            # Boost by activation and attention
            score *= item.activation_level * item.attention_weight

            if score > 0:
                memory = Memory(
                    id=f"wm_{id(item)}",
                    type=MemoryType.WORKING,
                    content=item.content,
                    timestamp=item.timestamp,
                    importance=item.relevance,
                )
                memories.append(memory)
                scores.append(score)

        # Sort by score
        sorted_pairs = sorted(zip(memories, scores), key=lambda x: x[1], reverse=True)
        memories = [m for m, _ in sorted_pairs[: query.limit]]
        scores = [s for _, s in sorted_pairs[: query.limit]]

        result = RetrievalResult(
            memories=memories,
            scores=scores,
            query_time_ms=(time.time() - start_time) * 1000,
            total_matches=len(memories),
        )

        self.stats.total_queries += 1

        return result

    def forget(self, memory_id: str) -> bool:
        """Remove from working memory."""
        # Working memory doesn't track by ID, so clear matching content
        self.clear()
        return True

    def consolidate(self) -> int:
        """Transfer important items to long-term memory."""
        # In a full system, would transfer to episodic/semantic memory
        consolidated = 0

        for item in list(self.buffer):
            if item.relevance > 0.7 and item.activation_level > 0.5:
                # Mark for transfer (would actually transfer in full system)
                consolidated += 1

        self.stats.total_consolidations += 1

        return consolidated

    def add(self, content: Any, relevance: float = 1.0) -> bool:
        """Add to working memory."""
        memory = self.store(content, relevance=relevance)
        return memory is not None

    def update_attention(self, attention_scores: List[float]):
        """Update attention weights."""
        if len(attention_scores) != len(self.buffer):
            return

        for i, score in enumerate(attention_scores):
            if i < len(self.buffer):
                self.buffer[i].attention_weight = score

        # Update focus to highest attention
        if self.buffer:
            self.focus = max(self.buffer, key=lambda x: x.attention_weight)

    def get_focused(self) -> Optional[Any]:
        """Get currently focused item."""
        if self.focus:
            return self.focus.content
        return None

    def rehearse(self):
        """Rehearse items to prevent decay."""
        current_time = time.time()

        for item in self.buffer:
            # Time-based decay
            time_delta = current_time - item.timestamp
            decay = np.exp(-0.1 * time_delta)

            # Update activation
            item.activation_level *= decay

            # Boost based on attention
            item.activation_level *= 1 + item.attention_weight * 0.1

            # Boost focused item
            if item == self.focus:
                item.activation_level *= 1.2

            # Clamp activation
            item.activation_level = np.clip(item.activation_level, 0, 1)

            # Remove if activation too low
            if item.activation_level < 0.1:
                self.buffer.remove(item)

    def add_to_phonological_loop(self, verbal_content: str):
        """Add verbal/acoustic information."""
        self.phonological_loop.append(
            {"content": verbal_content, "timestamp": time.time()}
        )

    def add_to_visuospatial_sketchpad(self, visual_content: Any):
        """Add visual/spatial information."""
        self.visuospatial_sketchpad.append(
            {"content": visual_content, "timestamp": time.time()}
        )

    def execute_task(self, task: Callable) -> Any:
        """Execute task using central executive."""
        self.task_queue.append(task)

        # Process immediately if no current task
        if self.current_task is None:
            return self._process_next_task()

        return None

    def _process_next_task(self) -> Any:
        """Process next task in queue."""
        if self.task_queue:
            self.current_task = self.task_queue.popleft()

            try:
                # Execute with working memory context
                context = {"buffer": list(self.buffer), "focus": self.focus}
                result = self.current_task(context)
            except Exception as e:
                logger.error(f"Task execution failed: {e}")
                result = None
            finally:
                self.current_task = None

            return result

        return None

    def _rehearsal_loop(self):
        """Background rehearsal loop."""
        while self._running:
            time.sleep(0.5)  # Rehearse every 500ms
            try:
                self.rehearse()

                # Decay phonological loop (faster decay ~2 seconds)
                current_time = time.time()
                self.phonological_loop = deque(
                    [
                        item
                        for item in self.phonological_loop
                        if current_time - item["timestamp"] < 2
                    ],
                    maxlen=7,
                )

                # Decay visuospatial sketchpad (slower decay)
                self.visuospatial_sketchpad = deque(
                    [
                        item
                        for item in self.visuospatial_sketchpad
                        if current_time - item["timestamp"] < 5
                    ],
                    maxlen=4,
                )

            except Exception as e:
                logger.error(f"Rehearsal error: {e}")

    def clear(self):
        """Clear working memory."""
        self.buffer.clear()
        self.focus = None
        self.attention_weights.clear()
        self.phonological_loop.clear()
        self.visuospatial_sketchpad.clear()
        self.task_queue.clear()

    def __del__(self):
        """Cleanup."""
        self.shutdown()
    
    def shutdown(self):
        """
        FIX #MEM-10: Clean shutdown of rehearsal thread.
        
        This should be called when the WorkingMemory is no longer needed
        to prevent thread resource leaks.
        """
        self._running = False
        if hasattr(self, 'rehearsal_thread') and self.rehearsal_thread.is_alive():
            self.rehearsal_thread.join(timeout=2.0)
            if self.rehearsal_thread.is_alive():
                logger.warning("WorkingMemory rehearsal thread did not stop in time")
