"""
Graphix Evolution Engine (Production-Ready)
============================================
Version: 2.0.0 - All issues fixed, stubs implemented
Tournament-based evolution of computation graphs with genetic algorithms.
"""

import asyncio
import copy
import hashlib
import json
import logging
import os
import random
import re
import threading
import time
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

try:
    from unified_runtime_core import get_runtime

    VULCAN_AVAILABLE = True
except ImportError:
    VULCAN_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Constants
MAX_CACHE_SIZE = 10000
CACHE_CLEANUP_THRESHOLD = 0.8  # Clean when 80% full


@dataclass
class Individual:
    """Represents an individual in the population."""

    graph: Dict[str, Any]
    fitness: float = 0.0
    generation: int = 0
    mutations: List[str] = field(default_factory=list)
    parent_ids: List[str] = field(default_factory=list)
    id: str = field(
        default_factory=lambda: hashlib.md5(str(time.time()).encode(), usedforsecurity=False).hexdigest()[:8]
    )
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheStatistics:
    """Cache performance statistics."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    current_size: int = 0
    max_size: int = MAX_CACHE_SIZE

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "current_size": self.current_size,
            "max_size": self.max_size,
            "hit_rate": self.hit_rate,
        }


class LRUCache:
    """LRU cache with size limit for fitness values."""

    def __init__(self, max_size: int = MAX_CACHE_SIZE):
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()
        self.stats = CacheStatistics(max_size=max_size)
        self.lock = threading.RLock()

    def get(self, key: str) -> Optional[float]:
        """Get value from cache."""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.stats.hits += 1
                return self.cache[key]

            self.stats.misses += 1
            return None

    def put(self, key: str, value: float):
        """Put value in cache with LRU eviction."""
        with self.lock:
            if key in self.cache:
                # Update existing
                self.cache.move_to_end(key)
                self.cache[key] = value
            else:
                # Add new entry
                if len(self.cache) >= self.max_size:
                    # Evict oldest
                    self.cache.popitem(last=False)
                    self.stats.evictions += 1

                self.cache[key] = value

            self.stats.current_size = len(self.cache)

    def clear(self):
        """Clear cache."""
        with self.lock:
            self.cache.clear()
            self.stats.current_size = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return self.stats.to_dict()


class EvolutionEngine:
    """
    Production-ready evolution engine with:
    - Real subgraph crossover
    - Async parallel fitness evaluation
    - Bounded LRU cache
    - Cache statistics
    - Diversity maintenance
    - Validated file I/O
    - Comprehensive error handling
    """

    # Security: Define allowed node types
    ALLOWED_NODE_TYPES = [
        "input",
        "output",
        "transform",
        "aggregate",
        "filter",
        "map",
        "reduce",
        "join",
        "split",
        "cache",
        "optimize",
        "normalize",
        "encode",
        "decode",
        "merge",
        "route",
    ]

    # Security: Resource limits
    MAX_NODES = 100
    MAX_EDGES = 500
    MAX_PARAM_LENGTH = 100
    MAX_STRING_LENGTH = 1000

    def __init__(
        self,
        population_size: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        tournament_size: int = 5,
        elitism_rate: float = 0.1,
        max_generations: int = 1000,
        max_nodes: int = 50,
        max_edges: int = 200,
        cache_size: int = MAX_CACHE_SIZE,
        diversity_threshold: float = 0.1,
        use_vulcan_fitness: bool = True,
    ):
        """Initialize evolution engine with genetic algorithm parameters."""
        # Validate and limit parameters
        self.population_size = min(max(10, population_size), 1000)
        self.mutation_rate = min(max(0.0, mutation_rate), 1.0)
        self.crossover_rate = min(max(0.0, crossover_rate), 1.0)
        self.tournament_size = min(max(2, tournament_size), 20)
        self.elitism_rate = min(max(0.0, elitism_rate), 0.5)
        self.max_generations = min(max(1, max_generations), 10000)
        self.max_nodes = min(max(3, max_nodes), self.MAX_NODES)
        self.max_edges = min(max(2, max_edges), self.MAX_EDGES)
        self.diversity_threshold = min(max(0.0, diversity_threshold), 1.0)
        self.use_vulcan_fitness = use_vulcan_fitness

        # State
        self.population: List[Individual] = []
        self.generation = 0
        self.best_individual: Optional[Individual] = None
        self.history = deque(maxlen=1000)

        # LRU cache with statistics
        self.fitness_cache = LRUCache(max_size=cache_size)

        # Thread safety
        self.lock = threading.RLock()

        # Mutation operators
        self.mutation_operators = {
            "add_node": self._mutate_add_node,
            "remove_node": self._mutate_remove_node,
            "modify_edge": self._mutate_modify_edge,
            "change_parameter": self._mutate_change_parameter,
            "swap_nodes": self._mutate_swap_nodes,
            "duplicate_subgraph": self._mutate_duplicate_subgraph,
        }

        # Crossover operators
        self.crossover_operators = {
            "single_point": self._crossover_single_point,
            "uniform": self._crossover_uniform,
            "subgraph": self._crossover_subgraph,
        }

        self.node_types = [
            t for t in self.ALLOWED_NODE_TYPES if t not in ["input", "output"]
        ]

        logger.info(
            f"EvolutionEngine initialized: pop_size={self.population_size}, "
            f"cache_size={cache_size}, diversity_threshold={diversity_threshold}"
        )

    def _sanitize_string(self, value: str, max_length: int = None) -> str:
        """Sanitize string values to prevent injection attacks."""
        if max_length is None:
            max_length = self.MAX_STRING_LENGTH

        # Remove dangerous characters
        sanitized = re.sub(r'[;`$()<>|&\\"\']', "", str(value))
        # Remove control characters
        sanitized = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", sanitized)
        # Limit length
        return sanitized[:max_length]

    def _sanitize_params(self, params: Dict) -> Dict:
        """Sanitize parameters to prevent injection and limit resources."""
        if not isinstance(params, dict):
            return {}

        safe_params = {}
        for key, value in params.items():
            # Sanitize key
            safe_key = self._sanitize_string(key, self.MAX_PARAM_LENGTH)

            # Sanitize value based on type
            if isinstance(value, bool):
                safe_params[safe_key] = value
            elif isinstance(value, (int, float)):
                # Limit numeric ranges
                if isinstance(value, int):
                    safe_params[safe_key] = max(-1000000, min(1000000, value))
                else:
                    safe_params[safe_key] = max(-1e6, min(1e6, value))
            elif isinstance(value, str):
                safe_params[safe_key] = self._sanitize_string(
                    value, self.MAX_PARAM_LENGTH
                )
            elif isinstance(value, list):
                # Limit list size and sanitize elements
                safe_list = []
                for item in value[:10]:  # Max 10 items
                    if isinstance(item, (int, float, bool)):
                        safe_list.append(item)
                    elif isinstance(item, str):
                        safe_list.append(self._sanitize_string(item, 50))
                safe_params[safe_key] = safe_list
            elif isinstance(value, dict):
                # Recursively sanitize nested dicts (one level only)
                if len(safe_params) < 20:  # Limit total params
                    safe_params[safe_key] = self._sanitize_params(value)
            # Ignore other types for safety

        return safe_params

    def _validate_node(self, node: Dict) -> bool:
        """Validate a node structure."""
        if not isinstance(node, dict):
            return False

        if "id" not in node or "type" not in node:
            return False

        # Validate node type
        if node["type"] not in self.ALLOWED_NODE_TYPES:
            return False

        # Validate and sanitize params if present
        if "params" in node:
            node["params"] = self._sanitize_params(node.get("params", {}))

        return True

    def _validate_edge(self, edge: Dict, node_ids: set) -> bool:
        """Validate an edge structure."""
        if not isinstance(edge, dict):
            return False

        if "source" not in edge or "target" not in edge:
            return False

        # Check that source and target exist
        if edge["source"] not in node_ids or edge["target"] not in node_ids:
            return False

        # Validate weight if present
        if "weight" in edge:
            if not isinstance(edge["weight"], (int, float)):
                edge["weight"] = 1.0
            else:
                edge["weight"] = max(0.0, min(1.0, edge["weight"]))

        return True

    def _validate_graph(self, graph: Dict) -> Dict:
        """Validate and sanitize a graph structure."""
        if not isinstance(graph, dict):
            return self._generate_minimal_graph()

        # Ensure required fields
        if "nodes" not in graph or not isinstance(graph["nodes"], list):
            graph["nodes"] = []
        if "edges" not in graph or not isinstance(graph["edges"], list):
            graph["edges"] = []

        # Limit and validate nodes
        validated_nodes = []
        node_ids = set()

        for node in graph["nodes"][: self.max_nodes]:
            if self._validate_node(node):
                node["id"] = self._sanitize_string(str(node["id"]), 50)
                if node["id"] not in node_ids:
                    validated_nodes.append(node)
                    node_ids.add(node["id"])

        # Ensure minimum nodes
        if len(validated_nodes) < 2:
            return self._generate_minimal_graph()

        # Limit and validate edges
        validated_edges = []
        edge_set = set()

        for edge in graph["edges"][: self.max_edges]:
            if self._validate_edge(edge, node_ids):
                edge_key = (edge["source"], edge["target"])
                if edge_key not in edge_set:
                    validated_edges.append(edge)
                    edge_set.add(edge_key)

        return {
            "grammar_version": "2.1.0",
            "nodes": validated_nodes,
            "edges": validated_edges,
            "metadata": self._sanitize_params(graph.get("metadata", {})),
        }

    def _generate_minimal_graph(self) -> Dict:
        """Generate a minimal valid graph."""
        return {
            "grammar_version": "2.1.0",
            "nodes": [
                {"id": "input", "type": "input", "params": {}},
                {"id": "output", "type": "output", "params": {}},
            ],
            "edges": [{"source": "input", "target": "output", "weight": 1.0}],
            "metadata": {},
        }

    def initialize_population(self, seed_graph: Optional[Dict[str, Any]] = None):
        """Initialize population with random or seeded graphs."""
        with self.lock:
            self.population = []

            if seed_graph:
                # Validate and use seed graph
                validated_seed = self._validate_graph(seed_graph)

                # Start with variations of seed graph
                for i in range(self.population_size):
                    individual = Individual(
                        graph=copy.deepcopy(validated_seed), generation=0
                    )
                    if i > 0:  # Mutate all except first
                        individual.graph = self._apply_random_mutation(individual.graph)
                        individual.mutations.append(f"initial_variation_{i}")
                    self.population.append(individual)
            else:
                # Generate random graphs
                for i in range(self.population_size):
                    graph = self._generate_random_graph()
                    self.population.append(Individual(graph=graph, generation=0))

            logger.info(
                f"Population initialized with {len(self.population)} individuals"
            )

    def evolve(
        self,
        fitness_function: Callable[[Dict], float],
        generations: Optional[int] = None,
    ) -> Individual:
        """Run synchronous evolution for specified generations."""
        generations = generations or self.max_generations
        generations = min(generations, self.max_generations)  # Enforce limit

        for gen in range(generations):
            # Evaluate fitness
            self._evaluate_population(fitness_function)

            # Select best
            with self.lock:
                self.best_individual = max(self.population, key=lambda x: x.fitness)

            # Log progress
            avg_fitness = np.mean([ind.fitness for ind in self.population])
            diversity = self._calculate_diversity()

            logger.info(
                f"Gen {gen}: Best={self.best_individual.fitness:.4f}, "
                f"Avg={avg_fitness:.4f}, Diversity={diversity:.4f}"
            )

            # Record history
            with self.lock:
                self.history.append(
                    {
                        "generation": gen,
                        "best_fitness": self.best_individual.fitness,
                        "avg_fitness": avg_fitness,
                        "diversity": diversity,
                        "cache_stats": self.fitness_cache.get_stats(),
                    }
                )

            # Check termination
            if self.best_individual.fitness >= 0.99:
                logger.info(f"Optimal solution found at generation {gen}")
                with self.lock:
                    self.generation = gen + 1
                break

            # Maintain diversity
            if diversity < self.diversity_threshold:
                logger.warning(
                    f"Low diversity ({diversity:.4f}), injecting random individuals"
                )
                self._inject_diversity()

            # Create next generation
            with self.lock:
                self.population = self._create_next_generation()
                self.generation = gen + 1

        return self.best_individual

    async def evolve_async(
        self,
        fitness_function: Callable[[Dict], float],
        generations: Optional[int] = None,
        max_workers: int = 4,
    ) -> Individual:
        """
        Run asynchronous evolution with parallel fitness evaluation.

        Args:
            fitness_function: Fitness evaluation function
            generations: Number of generations
            max_workers: Number of parallel workers

        Returns:
            Best individual found
        """
        generations = generations or self.max_generations
        generations = min(generations, self.max_generations)

        for gen in range(generations):
            # Parallel fitness evaluation
            await self._evaluate_population_async(fitness_function, max_workers)

            # Select best
            with self.lock:
                self.best_individual = max(self.population, key=lambda x: x.fitness)

            # Log progress
            avg_fitness = np.mean([ind.fitness for ind in self.population])
            diversity = self._calculate_diversity()

            logger.info(
                f"Gen {gen} (async): Best={self.best_individual.fitness:.4f}, "
                f"Avg={avg_fitness:.4f}, Diversity={diversity:.4f}"
            )

            # Record history
            with self.lock:
                self.history.append(
                    {
                        "generation": gen,
                        "best_fitness": self.best_individual.fitness,
                        "avg_fitness": avg_fitness,
                        "diversity": diversity,
                        "cache_stats": self.fitness_cache.get_stats(),
                    }
                )

            # Check termination
            if self.best_individual.fitness >= 0.99:
                logger.info(f"Optimal solution found at generation {gen}")
                with self.lock:
                    self.generation = gen + 1
                break

            # Maintain diversity
            if diversity < self.diversity_threshold:
                logger.warning(f"Low diversity, injecting random individuals")
                self._inject_diversity()

            # Create next generation
            with self.lock:
                self.population = self._create_next_generation()
                self.generation = gen + 1

        return self.best_individual

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
                    self.fitness_cache.put(graph_hash, individual.fitness)

                except Exception as e:
                    logger.warning(f"Fitness evaluation failed: {e}")
                    individual.fitness = 0.0

    async def _evaluate_population_async(
        self, fitness_function: Callable, max_workers: int
    ):
        """Asynchronous parallel fitness evaluation."""

        async def evaluate_individual(ind: Individual):
            """Evaluate single individual."""
            graph_hash = self._hash_graph(ind.graph)
            cached_fitness = self.fitness_cache.get(graph_hash)

            if cached_fitness is not None:
                ind.fitness = cached_fitness
            else:
                try:
                    validated = self._validate_graph(ind.graph)
                    # Run in thread pool to avoid blocking
                    loop = asyncio.get_event_loop()
                    fitness = await loop.run_in_executor(
                        None, fitness_function, validated
                    )
                    ind.fitness = max(0.0, min(1.0, fitness))
                    self.fitness_cache.put(graph_hash, ind.fitness)
                except Exception as e:
                    logger.warning(f"Async fitness evaluation failed: {e}")
                    ind.fitness = 0.0

        # Evaluate all individuals in parallel
        tasks = [evaluate_individual(ind) for ind in self.population]
        await asyncio.gather(*tasks)

    def _create_next_generation(self) -> List[Individual]:
        """Create next generation through selection, crossover, and mutation."""
        next_generation = []

        # Elitism - keep best individuals
        elite_count = int(self.population_size * self.elitism_rate)
        elite = sorted(self.population, key=lambda x: x.fitness, reverse=True)[
            :elite_count
        ]
        next_generation.extend([copy.deepcopy(ind) for ind in elite])

        # Generate rest through crossover and mutation
        while len(next_generation) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()

            # Crossover
            if random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)

            # Mutation
            if random.random() < self.mutation_rate:
                child1.graph = self._apply_random_mutation(child1.graph)
                child1.mutations.append(f"gen_{self.generation}_mutation")

            if random.random() < self.mutation_rate:
                child2.graph = self._apply_random_mutation(child2.graph)
                child2.mutations.append(f"gen_{self.generation}_mutation")

            # Update generation
            child1.generation = self.generation + 1
            child2.generation = self.generation + 1

            # Validate graphs
            child1.graph = self._validate_graph(child1.graph)
            child2.graph = self._validate_graph(child2.graph)

            next_generation.extend([child1, child2])

        return next_generation[: self.population_size]

    def _tournament_selection(self) -> Individual:
        """Select individual through tournament."""
        tournament_size = min(self.tournament_size, len(self.population))
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=lambda x: x.fitness)

    def _crossover(
        self, parent1: Individual, parent2: Individual
    ) -> Tuple[Individual, Individual]:
        """Perform crossover between two parents."""
        operator = random.choice(list(self.crossover_operators.values()))
        try:
            graph1, graph2 = operator(parent1.graph, parent2.graph)

            # Validate resulting graphs
            graph1 = self._validate_graph(graph1)
            graph2 = self._validate_graph(graph2)

            child1 = Individual(
                graph=graph1,
                generation=self.generation + 1,
                parent_ids=[parent1.id, parent2.id],
            )
            child2 = Individual(
                graph=graph2,
                generation=self.generation + 1,
                parent_ids=[parent1.id, parent2.id],
            )

            return child1, child2
        except Exception as e:
            logger.warning(f"Crossover failed: {e}")
            return copy.deepcopy(parent1), copy.deepcopy(parent2)

    def _apply_random_mutation(self, graph: Dict) -> Dict:
        """Apply random mutation to graph."""
        operator = random.choice(list(self.mutation_operators.values()))
        try:
            mutated = operator(copy.deepcopy(graph))
            return self._validate_graph(mutated)
        except Exception as e:
            logger.warning(f"Mutation failed: {e}")
            return graph

    def _mutate_add_node(self, graph: Dict) -> Dict:
        """Add a random node to the graph."""
        if "nodes" not in graph:
            graph["nodes"] = []

        # Check node limit
        if len(graph["nodes"]) >= self.max_nodes:
            return graph

        new_node = {
            "id": f"node_{len(graph['nodes'])}_{random.randint(1000, 9999)}",
            "type": random.choice(self.node_types),
            "params": self._sanitize_params({"value": random.random()}),
        }

        graph["nodes"].append(new_node)

        # Add random edge
        if len(graph["nodes"]) > 1:
            if "edges" not in graph:
                graph["edges"] = []

            # Check edge limit
            if len(graph["edges"]) < self.max_edges:
                source = random.choice(graph["nodes"][:-1])
                graph["edges"].append(
                    {
                        "source": source["id"],
                        "target": new_node["id"],
                        "weight": random.random(),
                    }
                )

        return graph

    def _mutate_remove_node(self, graph: Dict) -> Dict:
        """Remove a random node from the graph."""
        if "nodes" not in graph or len(graph.get("nodes", [])) <= 2:
            return graph

        # Don't remove input/output nodes
        removable = [
            n for n in graph["nodes"] if n.get("type") not in ["input", "output"]
        ]

        if removable:
            node_to_remove = random.choice(removable)
            graph["nodes"].remove(node_to_remove)

            # Remove associated edges
            if "edges" in graph:
                graph["edges"] = [
                    e
                    for e in graph["edges"]
                    if e["source"] != node_to_remove["id"]
                    and e["target"] != node_to_remove["id"]
                ]

        return graph

    def _mutate_modify_edge(self, graph: Dict) -> Dict:
        """Modify a random edge weight."""
        if "edges" not in graph or not graph["edges"]:
            return graph

        edge = random.choice(graph["edges"])
        edge["weight"] = random.random()

        return graph

    def _mutate_change_parameter(self, graph: Dict) -> Dict:
        """Change a random node parameter."""
        if "nodes" not in graph or not graph["nodes"]:
            return graph

        node = random.choice(graph["nodes"])
        if "params" not in node:
            node["params"] = {}

        # Modify or add parameter with safe values
        param_name = random.choice(["value", "threshold", "rate", "size"])
        if param_name in ["value", "threshold", "rate"]:
            node["params"][param_name] = random.random()
        elif param_name == "size":
            node["params"][param_name] = random.randint(1, 100)

        # Sanitize all params
        node["params"] = self._sanitize_params(node["params"])

        return graph

    def _mutate_swap_nodes(self, graph: Dict) -> Dict:
        """Swap two random nodes' types."""
        if "nodes" not in graph or len(graph["nodes"]) < 2:
            return graph

        swappable = [
            n for n in graph["nodes"] if n.get("type") not in ["input", "output"]
        ]

        if len(swappable) >= 2:
            node1, node2 = random.sample(swappable, 2)
            # Only swap between allowed types
            if node1["type"] in self.node_types and node2["type"] in self.node_types:
                node1["type"], node2["type"] = node2["type"], node1["type"]

        return graph

    def _mutate_duplicate_subgraph(self, graph: Dict) -> Dict:
        """Duplicate a random subgraph."""
        if "nodes" not in graph or len(graph["nodes"]) < 3:
            return graph

        # Check limits
        if len(graph["nodes"]) >= self.max_nodes - 3:
            return graph

        # Select random subset of nodes
        duplicable = [
            n for n in graph["nodes"] if n.get("type") not in ["input", "output"]
        ]

        if not duplicable:
            return graph

        num_nodes = random.randint(1, min(3, len(duplicable)))
        subgraph_nodes = random.sample(duplicable, num_nodes)

        # Duplicate nodes with new IDs
        id_mapping = {}
        for node in subgraph_nodes:
            if len(graph["nodes"]) >= self.max_nodes:
                break

            new_node = copy.deepcopy(node)
            new_id = f"{node['id']}_dup_{random.randint(1000, 9999)}"
            new_node["id"] = new_id
            new_node["params"] = self._sanitize_params(new_node.get("params", {}))
            id_mapping[node["id"]] = new_id
            graph["nodes"].append(new_node)

        # Duplicate edges within subgraph
        if "edges" in graph and id_mapping:
            edges_to_add = []
            for edge in graph["edges"]:
                if len(graph["edges"]) + len(edges_to_add) >= self.max_edges:
                    break

                if edge["source"] in id_mapping and edge["target"] in id_mapping:
                    new_edge = {
                        "source": id_mapping[edge["source"]],
                        "target": id_mapping[edge["target"]],
                        "weight": edge.get("weight", 1.0),
                    }
                    edges_to_add.append(new_edge)

            graph["edges"].extend(edges_to_add)

        return graph

    def _crossover_single_point(self, graph1: Dict, graph2: Dict) -> Tuple[Dict, Dict]:
        """Single point crossover."""
        if "nodes" not in graph1 or "nodes" not in graph2:
            return graph1, graph2

        if not graph1["nodes"] or not graph2["nodes"]:
            return graph1, graph2

        point = random.randint(1, min(len(graph1["nodes"]), len(graph2["nodes"])) - 1)

        child1 = copy.deepcopy(graph1)
        child2 = copy.deepcopy(graph2)

        # Swap nodes after crossover point
        child1["nodes"] = (
            graph1["nodes"][:point] + graph2["nodes"][point : self.max_nodes]
        )
        child2["nodes"] = (
            graph2["nodes"][:point] + graph1["nodes"][point : self.max_nodes]
        )

        # Update edges to match new node sets
        child1_ids = {n["id"] for n in child1["nodes"]}
        child2_ids = {n["id"] for n in child2["nodes"]}

        if "edges" in child1:
            child1["edges"] = [
                e
                for e in child1["edges"]
                if e["source"] in child1_ids and e["target"] in child1_ids
            ][: self.max_edges]

        if "edges" in child2:
            child2["edges"] = [
                e
                for e in child2["edges"]
                if e["source"] in child2_ids and e["target"] in child2_ids
            ][: self.max_edges]

        return child1, child2

    def _crossover_uniform(self, graph1: Dict, graph2: Dict) -> Tuple[Dict, Dict]:
        """Uniform crossover."""
        child1 = copy.deepcopy(graph1)
        child2 = copy.deepcopy(graph2)

        if "nodes" in graph1 and "nodes" in graph2:
            min_nodes = min(len(graph1["nodes"]), len(graph2["nodes"]))
            for i in range(min_nodes):
                if random.random() < 0.5:
                    # Swap nodes
                    temp = child1["nodes"][i]
                    child1["nodes"][i] = copy.deepcopy(child2["nodes"][i])
                    child2["nodes"][i] = copy.deepcopy(temp)

        # Fix edges after node swapping
        child1_ids = {n["id"] for n in child1.get("nodes", [])}
        child2_ids = {n["id"] for n in child2.get("nodes", [])}

        if "edges" in child1:
            child1["edges"] = [
                e
                for e in child1["edges"]
                if e["source"] in child1_ids and e["target"] in child1_ids
            ][: self.max_edges]

        if "edges" in child2:
            child2["edges"] = [
                e
                for e in child2["edges"]
                if e["source"] in child2_ids and e["target"] in child2_ids
            ][: self.max_edges]

        return child1, child2

    def _crossover_subgraph(self, graph1: Dict, graph2: Dict) -> Tuple[Dict, Dict]:
        """
        Real subgraph crossover - exchange connected subgraphs between parents.

        This finds connected components and swaps them between parents.
        """
        if "nodes" not in graph1 or "nodes" not in graph2:
            return graph1, graph2

        if not graph1["nodes"] or not graph2["nodes"]:
            return graph1, graph2

        try:
            # Find connected subgraphs in both parents
            subgraphs1 = self._find_connected_subgraphs(graph1)
            subgraphs2 = self._find_connected_subgraphs(graph2)

            if not subgraphs1 or not subgraphs2:
                # Fallback to uniform crossover
                return self._crossover_uniform(graph1, graph2)

            # Select random subgraphs to exchange
            sub1 = random.choice(subgraphs1)
            sub2 = random.choice(subgraphs2)

            # Build children by exchanging subgraphs
            child1 = self._exchange_subgraph(graph1, sub1, graph2, sub2)
            child2 = self._exchange_subgraph(graph2, sub2, graph1, sub1)

            return child1, child2

        except Exception as e:
            logger.warning(f"Subgraph crossover failed: {e}, using uniform")
            return self._crossover_uniform(graph1, graph2)

    def _find_connected_subgraphs(self, graph: Dict) -> List[Set[str]]:
        """
        Find connected components in graph.

        Returns:
            List of sets, each containing node IDs in a component
        """
        if "nodes" not in graph or "edges" not in graph:
            return []

        # Build adjacency list
        adjacency: Dict[str, Set[str]] = defaultdict(set)
        node_ids = {n["id"] for n in graph["nodes"]}

        for edge in graph["edges"]:
            source = edge["source"]
            target = edge["target"]
            if source in node_ids and target in node_ids:
                adjacency[source].add(target)
                adjacency[target].add(source)

        # Find connected components using DFS
        visited = set()
        components = []

        for node_id in node_ids:
            if node_id not in visited:
                component = set()
                stack = [node_id]

                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        component.add(current)
                        stack.extend(adjacency[current] - visited)

                # Only include non-trivial components with transformable nodes
                non_io_nodes = [
                    nid
                    for nid in component
                    if any(
                        n["id"] == nid and n.get("type") not in ["input", "output"]
                        for n in graph["nodes"]
                    )
                ]

                if len(non_io_nodes) >= 1:
                    components.append(component)

        return components

    def _exchange_subgraph(
        self, graph1: Dict, subgraph1: Set[str], graph2: Dict, subgraph2: Set[str]
    ) -> Dict:
        """
        Create child by replacing subgraph1 in graph1 with subgraph2 from graph2.
        """
        child = copy.deepcopy(graph1)

        # Remove nodes from subgraph1
        child["nodes"] = [n for n in child["nodes"] if n["id"] not in subgraph1]

        # Add nodes from subgraph2
        nodes_to_add = [
            copy.deepcopy(n) for n in graph2["nodes"] if n["id"] in subgraph2
        ]

        # Check size limits
        if len(child["nodes"]) + len(nodes_to_add) <= self.max_nodes:
            child["nodes"].extend(nodes_to_add)
        else:
            # Add as many as possible
            available_space = self.max_nodes - len(child["nodes"])
            child["nodes"].extend(nodes_to_add[:available_space])

        # Update edges
        child_node_ids = {n["id"] for n in child["nodes"]}

        # Keep edges not involving subgraph1
        child["edges"] = [
            e
            for e in child.get("edges", [])
            if e["source"] in child_node_ids and e["target"] in child_node_ids
        ]

        # Add edges from subgraph2
        edges_to_add = [
            copy.deepcopy(e)
            for e in graph2.get("edges", [])
            if e["source"] in subgraph2
            and e["target"] in subgraph2
            and e["source"] in child_node_ids
            and e["target"] in child_node_ids
        ]

        if len(child["edges"]) + len(edges_to_add) <= self.max_edges:
            child["edges"].extend(edges_to_add)
        else:
            available_space = self.max_edges - len(child["edges"])
            child["edges"].extend(edges_to_add[:available_space])

        return child

    def _generate_random_graph(self) -> Dict:
        """Generate a random graph SAFELY with validation."""
        num_nodes = random.randint(3, min(10, self.max_nodes))
        nodes = []
        edges = []

        # Always have input and output
        nodes.append({"id": "input", "type": "input", "params": {}})

        # Add random intermediate nodes
        for i in range(num_nodes - 2):
            node_type = random.choice(self.node_types)
            params = {
                "value": random.random(),
                "threshold": random.random(),
                "active": random.choice([True, False]),
            }
            nodes.append(
                {
                    "id": f"node_{i}",
                    "type": node_type,
                    "params": self._sanitize_params(params),
                }
            )

        nodes.append({"id": "output", "type": "output", "params": {}})

        # Create edges ensuring connectivity
        for i in range(len(nodes) - 1):
            edges.append(
                {
                    "source": nodes[i]["id"],
                    "target": nodes[i + 1]["id"],
                    "weight": random.random(),
                }
            )

        # Add some random extra edges
        extra_edges = min(random.randint(0, num_nodes), self.max_edges - len(edges))
        for _ in range(extra_edges):
            source = random.choice(nodes[:-1])
            target = random.choice(nodes[1:])
            if source["id"] != target["id"]:
                # Check if edge already exists
                edge_exists = any(
                    e["source"] == source["id"] and e["target"] == target["id"]
                    for e in edges
                )
                if not edge_exists:
                    edges.append(
                        {
                            "source": source["id"],
                            "target": target["id"],
                            "weight": random.random(),
                        }
                    )

        graph = {
            "grammar_version": "2.1.0",
            "nodes": nodes,
            "edges": edges[: self.max_edges],
            "metadata": {"generated": True, "timestamp": time.time()},
        }

        return self._validate_graph(graph)

    def _hash_graph(self, graph: Dict) -> str:
        """Create hash of graph for caching."""
        # Sort for consistent hashing
        graph_str = json.dumps(graph, sort_keys=True)
        return hashlib.sha256(graph_str.encode()).hexdigest()

    def _calculate_diversity(self) -> float:
        """Calculate population diversity."""
        if len(self.population) < 2:
            return 0.0

        # Calculate fitness variance
        fitnesses = [ind.fitness for ind in self.population]
        diversity = float(np.std(fitnesses))

        # Also consider structural diversity
        unique_hashes = set()
        for ind in self.population:
            graph_hash = self._hash_graph(ind.graph)
            unique_hashes.add(graph_hash)

        structural_diversity = len(unique_hashes) / len(self.population)

        # Combined diversity score
        return (diversity + structural_diversity) / 2

    def _inject_diversity(self, fraction: float = 0.2):
        """
        Inject diversity by replacing worst individuals with random ones.

        Args:
            fraction: Fraction of population to replace
        """
        with self.lock:
            num_to_replace = int(len(self.population) * fraction)

            # Sort by fitness
            sorted_pop = sorted(self.population, key=lambda x: x.fitness)

            # Replace worst individuals
            for i in range(num_to_replace):
                new_graph = self._generate_random_graph()
                sorted_pop[i] = Individual(
                    graph=new_graph,
                    generation=self.generation,
                    metadata={"injected_for_diversity": True},
                )

            self.population = sorted_pop

            logger.info(f"Injected {num_to_replace} random individuals for diversity")

    def clear_cache(self):
        """Clear fitness cache and reset statistics."""
        self.fitness_cache.clear()
        logger.info("Fitness cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.fitness_cache.get_stats()

    def get_statistics(self) -> Dict[str, Any]:
        """Get evolution statistics."""
        with self.lock:
            return {
                "generation": self.generation,
                "best_fitness": self.best_individual.fitness
                if self.best_individual
                else 0,
                "population_size": len(self.population),
                "diversity": self._calculate_diversity(),
                "cache_stats": self.get_cache_stats(),
                "history": list(self.history),
            }

    def save_population(self, filepath: str):
        """Save current population to file with validation."""
        try:
            # Validate write permissions
            path = Path(filepath)

            # Check directory exists and is writable
            if not path.parent.exists():
                path.parent.mkdir(parents=True, exist_ok=True)

            if path.exists() and not os.access(path, os.W_OK):
                raise PermissionError(f"No write permission for {filepath}")

            if not path.exists() and not os.access(path.parent, os.W_OK):
                raise PermissionError(
                    f"No write permission for directory {path.parent}"
                )

            # Prepare data
            with self.lock:
                population_data = {
                    "generation": self.generation,
                    "population": [
                        {
                            "graph": ind.graph,
                            "fitness": ind.fitness,
                            "generation": ind.generation,
                            "mutations": ind.mutations,
                            "parent_ids": ind.parent_ids,
                            "id": ind.id,
                            "metadata": ind.metadata,
                        }
                        for ind in self.population
                    ],
                    "best_individual": {
                        "graph": self.best_individual.graph,
                        "fitness": self.best_individual.fitness,
                        "generation": self.best_individual.generation,
                        "id": self.best_individual.id,
                        "metadata": self.best_individual.metadata,
                    }
                    if self.best_individual
                    else None,
                    "config": {
                        "population_size": self.population_size,
                        "mutation_rate": self.mutation_rate,
                        "crossover_rate": self.crossover_rate,
                        "tournament_size": self.tournament_size,
                        "elitism_rate": self.elitism_rate,
                        "max_nodes": self.max_nodes,
                        "max_edges": self.max_edges,
                        "diversity_threshold": self.diversity_threshold,
                    },
                    "cache_stats": self.get_cache_stats(),
                    "timestamp": datetime.utcnow().isoformat(),
                }

            # Write to file
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(population_data, f, indent=2)

            logger.info(f"Saved population to {filepath}")

        except PermissionError as e:
            logger.error(f"Permission error saving population: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to save population: {e}")
            raise

    def load_population(self, filepath: str):
        """
        Load population from file with error recovery.

        Keeps existing population on error instead of initializing empty.
        """
        try:
            # Validate read permissions
            path = Path(filepath)

            if not path.exists():
                raise FileNotFoundError(f"Population file not found: {filepath}")

            if not os.access(path, os.R_OK):
                raise PermissionError(f"No read permission for {filepath}")

            # Read file
            with open(filepath, "r", encoding="utf-8") as f:
                population_data = json.load(f)

            # Store old population for recovery
            with self.lock:
                old_population = self.population.copy()
                old_generation = self.generation
                old_best = self.best_individual

                try:
                    # Load new population
                    self.generation = population_data["generation"]
                    self.population = []

                    for ind_data in population_data["population"]:
                        ind = Individual(
                            graph=self._validate_graph(ind_data["graph"]),
                            fitness=ind_data["fitness"],
                            generation=ind_data["generation"],
                            mutations=ind_data.get("mutations", []),
                            parent_ids=ind_data.get("parent_ids", []),
                            id=ind_data.get(
                                "id",
                                hashlib.md5(str(time.time()).encode(), usedforsecurity=False).hexdigest()[:8],
                            ),
                            metadata=ind_data.get("metadata", {}),
                        )
                        self.population.append(ind)

                    # Load best individual
                    if population_data.get("best_individual"):
                        best_data = population_data["best_individual"]
                        self.best_individual = Individual(
                            graph=self._validate_graph(best_data["graph"]),
                            fitness=best_data["fitness"],
                            generation=best_data["generation"],
                            id=best_data.get(
                                "id",
                                hashlib.md5(str(time.time()).encode(), usedforsecurity=False).hexdigest()[:8],
                            ),
                            metadata=best_data.get("metadata", {}),
                        )

                    # Load config if present
                    if "config" in population_data:
                        config = population_data["config"]
                        self.population_size = config.get(
                            "population_size", self.population_size
                        )
                        self.mutation_rate = config.get(
                            "mutation_rate", self.mutation_rate
                        )
                        self.crossover_rate = config.get(
                            "crossover_rate", self.crossover_rate
                        )
                        self.tournament_size = config.get(
                            "tournament_size", self.tournament_size
                        )
                        self.elitism_rate = config.get(
                            "elitism_rate", self.elitism_rate
                        )
                        self.max_nodes = config.get("max_nodes", self.max_nodes)
                        self.max_edges = config.get("max_edges", self.max_edges)
                        self.diversity_threshold = config.get(
                            "diversity_threshold", self.diversity_threshold
                        )

                    logger.info(f"Loaded population from {filepath}")

                except Exception as e:
                    # Restore old population on error
                    logger.error(
                        f"Error loading population data, restoring previous state: {e}"
                    )
                    self.population = old_population
                    self.generation = old_generation
                    self.best_individual = old_best
                    raise

        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            raise
        except PermissionError as e:
            logger.error(f"Permission error: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load population: {e}")
            raise


# Demo and testing
if __name__ == "__main__":
    import copy

    print("=" * 60)
    print("Evolution Engine - Production Demo")
    print("=" * 60)

    # Simple fitness function
    def fitness_fn(graph: Dict) -> float:
        """Simple fitness: reward more nodes and edges."""
        nodes = len(graph.get("nodes", []))
        edges = len(graph.get("edges", []))
        return min(1.0, (nodes + edges) / 20.0)

    # Test 1: Basic evolution
    print("\n1. Basic Synchronous Evolution")
    engine = EvolutionEngine(population_size=20, max_generations=10, cache_size=100)

    engine.initialize_population()
    best = engine.evolve(fitness_fn, generations=5)

    print(f"   Best fitness: {best.fitness:.4f}")
    print(f"   Best graph nodes: {len(best.graph['nodes'])}")
    print(f"   Cache stats: {engine.get_cache_stats()}")

    # Test 2: Async evolution
    print("\n2. Asynchronous Evolution")

    async def run_async():
        engine2 = EvolutionEngine(population_size=20, max_generations=10)
        engine2.initialize_population()
        best = await engine2.evolve_async(fitness_fn, generations=5, max_workers=2)
        return best

    best_async = asyncio.run(run_async())
    print(f"   Best fitness (async): {best_async.fitness:.4f}")

    # Test 3: Subgraph crossover
    print("\n3. Subgraph Crossover Test")

    # Create two test graphs
    graph1 = {
        "nodes": [
            {"id": "input", "type": "input"},
            {"id": "n1", "type": "transform"},
            {"id": "n2", "type": "filter"},
            {"id": "output", "type": "output"},
        ],
        "edges": [
            {"source": "input", "target": "n1"},
            {"source": "n1", "target": "n2"},
            {"source": "n2", "target": "output"},
        ],
    }

    graph2 = {
        "nodes": [
            {"id": "input", "type": "input"},
            {"id": "x1", "type": "map"},
            {"id": "x2", "type": "reduce"},
            {"id": "output", "type": "output"},
        ],
        "edges": [
            {"source": "input", "target": "x1"},
            {"source": "x1", "target": "x2"},
            {"source": "x2", "target": "output"},
        ],
    }

    child1, child2 = engine._crossover_subgraph(graph1, graph2)
    print(f"   Parent 1 nodes: {[n['id'] for n in graph1['nodes']]}")
    print(f"   Parent 2 nodes: {[n['id'] for n in graph2['nodes']]}")
    print(f"   Child 1 nodes: {[n['id'] for n in child1['nodes']]}")
    print(f"   Child 2 nodes: {[n['id'] for n in child2['nodes']]}")

    # Test 4: Cache management
    print("\n4. Cache Management")

    stats_before = engine.get_cache_stats()
    print(f"   Cache before clear: {stats_before}")

    engine.clear_cache()

    stats_after = engine.get_cache_stats()
    print(f"   Cache after clear: {stats_after}")

    # Test 5: File I/O
    print("\n5. File I/O with Validation")

    try:
        engine.save_population("test_population.json")
        print("   Save: SUCCESS")

        engine2 = EvolutionEngine(population_size=20)
        engine2.load_population("test_population.json")
        print("   Load: SUCCESS")
        print(f"   Loaded generation: {engine2.generation}")

        # Cleanup
        os.remove("test_population.json")
    except Exception as e:
        print(f"   File I/O error: {e}")

    # Test 6: Diversity maintenance
    print("\n6. Diversity Maintenance")

    engine3 = EvolutionEngine(population_size=30, diversity_threshold=0.2)
    engine3.initialize_population()

    # Force low diversity by making all individuals the same
    same_graph = engine3.population[0].graph
    for ind in engine3.population:
        ind.graph = copy.deepcopy(same_graph)

    diversity_before = engine3._calculate_diversity()
    print(f"   Diversity before injection: {diversity_before:.4f}")

    engine3._inject_diversity(fraction=0.3)

    diversity_after = engine3._calculate_diversity()
    print(f"   Diversity after injection: {diversity_after:.4f}")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
