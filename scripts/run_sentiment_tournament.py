# ====================================================================
# SCRIPT PROLOGUE:
# Add the project root directory to the Python path.
# This allows the script to import modules from 'src' and 'specs'.
# ====================================================================
import sys
from pathlib import Path

# Get the directory of the current script (scripts/) and go up one level to the project root (D:\Graphix)
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
# ====================================================================
#!/usr/bin/env python3
"""
run_sentiment_tournament.py — Graphix GA/Tournament driver (offline/online)

Offline mode: uses LanguageEvolutionRegistry for proposals/votes/validation and
simulates runtime metrics. Online mode: uses GraphixClient to submit/execute.

Version: 2.0.0 - Production-ready with all bug fixes
"""

import argparse
import asyncio
import copy
import hashlib
import json
import logging
import os
import random
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse


# =========================
# Configuration
# =========================
@dataclass
class GAConfig:
    """Configuration for genetic algorithm parameters."""

    # Fitness function coefficients (documented for reproducibility)
    base_tokens: float = 120.0
    tokens_per_node: float = 7.0
    base_cycles: float = 4500.0
    cycles_per_node: float = 25.0
    base_accuracy: float = 0.96
    accuracy_drop_per_node: float = 0.006

    # Fitness weighting (mirrors classification fitness intent)
    alpha_tokens: float = 1e-4
    beta_cycles_per_k: float = 1e-2

    # GA parameters
    mutation_rate: float = 0.1  # 10% mutation probability
    crossover_rate: float = 0.8  # 80% crossover probability
    max_champions: int = 50  # Maximum champions to keep in memory

    # Early stopping
    convergence_window: int = 10  # Generations to check for convergence
    convergence_threshold: float = 0.001  # Fitness change threshold

    # Validation limits
    max_graph_size_mb: float = 10.0  # Maximum graph size in MB
    max_nodes: int = 1000  # Maximum nodes per graph
    max_edges: int = 5000  # Maximum edges per graph


# =========================
# Default Paths
# =========================
DEFAULT_GRAPHS_DIR = Path("graphs")
DEFAULT_WORKLOADS_DIR = DEFAULT_GRAPHS_DIR / "workloads"
DEFAULT_SENTIMENT_GRAPH = DEFAULT_GRAPHS_DIR / "sentiment_3d.json"
DEFAULT_EVAL_WORKLOAD = DEFAULT_WORKLOADS_DIR / "sentiment_eval.json"
DEFAULT_CHAMPIONS_OUT = Path("evolution_champions")
DEFAULT_CHECKPOINT_DIR = Path("checkpoints")

# =========================
# Logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger(__name__)

# =========================
# Optional imports with graceful degradation
# =========================
try:
    from specs.formal_grammar.language_evolution_registry import (
        DevelopmentKMS,
        InMemoryBackend,
        LanguageEvolutionRegistry,
    )

    HAS_REGISTRY = True
except ImportError:
    HAS_REGISTRY = False
    log.warning("LanguageEvolutionRegistry not available, offline mode disabled")

try:
    # FIXED: Import AgentInterface instead of the non-existent GraphixClient
    from src.agent_interface import AgentInterface, ConnectionConfig

    HAS_SDK = True
except ImportError:
    AgentInterface, ConnectionConfig = None, None
    HAS_SDK = False
    log.warning("AgentInterface not available, online mode disabled")


# =========================
# Types
# =========================
@dataclass
class Metrics:
    """Metrics for graph evaluation."""

    accuracy: float
    tokens: float
    cycles: float
    score: float


@dataclass
class GraphCandidate:
    """A candidate graph with cached metrics."""

    graph: Dict[str, Any]
    metrics: Optional[Metrics] = None
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)


# =========================
# Validation Functions
# =========================
class GraphValidator:
    """Validates graph structure and content."""

    def __init__(self, config: GAConfig):
        self.config = config

    def validate_graph(self, graph: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate the structure of a Graphix IR graph."""
        if not isinstance(graph, dict):
            return False, "Graph must be a dictionary"

        nodes = graph.get("nodes")
        if not isinstance(nodes, list):
            return False, "'nodes' must be a list"

        # Validate nodes and collect IDs
        node_id_set = set()
        for i, node in enumerate(nodes):
            if not isinstance(node, dict) or "id" not in node:
                return False, f"Node at index {i} is malformed or missing 'id'"
            node_id_set.add(node["id"])

        edges = graph.get("edges")
        if not isinstance(edges, list):
            return False, "'edges' must be a list"

        # Validate edges
        for i, edge in enumerate(edges):
            if not isinstance(edge, dict) or "from" not in edge or "to" not in edge:
                return False, f"Edge at index {i} is malformed (missing from/to)"

            # --- START OF THE FIX ---
            # Handle both simple string and complex dictionary references
            from_ref = edge["from"]
            to_ref = edge["to"]

            source_id = from_ref["node"] if isinstance(from_ref, dict) else from_ref
            target_id = to_ref["node"] if isinstance(to_ref, dict) else to_ref
            # --- END OF THE FIX ---

            if source_id not in node_id_set:
                return False, f"Edge {i} has invalid 'from' reference: {source_id}"

            if target_id not in node_id_set:
                return False, f"Edge {i} has invalid 'to' reference: {target_id}"

        return True, None


class PathValidator:
    """Validates file paths for security."""

    @staticmethod
    def validate_path(path: Path, base_dir: Optional[Path] = None) -> Tuple[bool, str]:
        """
        Validate that path is safe to use.

        Args:
            path: Path to validate
            base_dir: Optional base directory to restrict to

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Resolve to absolute path
            abs_path = path.resolve()

            # Check if file exists
            if not abs_path.exists():
                return False, f"Path does not exist: {path}"

            # Check if within base directory
            if base_dir:
                abs_base = base_dir.resolve()
                try:
                    abs_path.relative_to(abs_base)
                except ValueError:
                    return False, f"Path outside allowed directory: {path}"

            # Check file size
            if abs_path.is_file():
                size_mb = abs_path.stat().st_size / (1024 * 1024)
                if size_mb > 100:  # 100MB limit
                    return False, f"File too large: {size_mb:.2f}MB"

            return True, ""
        except Exception as e:
            return False, f"Path validation error: {e}"


# =========================
# Fitness Evaluation
# =========================
class FitnessEvaluator:
    """Evaluates graph fitness with caching."""

    def __init__(self, config: GAConfig):
        self.config = config
        self.cache: Dict[str, Metrics] = {}

    def _compute_graph_hash(self, graph: Dict[str, Any]) -> str:
        """Compute deterministic hash of graph."""
        serialized = json.dumps(graph, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def evaluate(self, graph: Dict[str, Any]) -> Metrics:
        """
        Evaluate graph fitness using heuristic metrics.
        Caches results to avoid re-computation.

        Args:
            graph: Graph to evaluate

        Returns:
            Metrics with accuracy, tokens, cycles, and score
        """
        # Check cache
        graph_hash = self._compute_graph_hash(graph)
        if graph_hash in self.cache:
            return self.cache[graph_hash]

        # Compute metrics based on graph structure
        node_count = len(graph.get("nodes", []))
        edge_count = len(graph.get("edges", []))

        # Token usage (input + processing + output)
        tokens = self.config.base_tokens + (node_count * self.config.tokens_per_node)

        # Compute cycles (execution time proxy)
        cycles = self.config.base_cycles + (node_count * self.config.cycles_per_node)

        # Estimate accuracy (decreases with complexity)
        accuracy = max(
            0.0,
            min(
                1.0,
                self.config.base_accuracy
                - (node_count * self.config.accuracy_drop_per_node),
            ),
        )

        # Composite fitness score (maximize accuracy, minimize tokens and cycles)
        score = (
            accuracy
            - (tokens * self.config.alpha_tokens)
            - (cycles / 1000.0 * self.config.beta_cycles_per_k)
        )

        metrics = Metrics(accuracy=accuracy, tokens=tokens, cycles=cycles, score=score)

        # Cache result
        self.cache[graph_hash] = metrics

        log.debug(
            f"Fitness for {graph['id']}: score={score:.4f}, acc={accuracy:.3f}, "
            f"tokens={tokens:.0f}, cycles={cycles:.0f}"
        )

        return metrics

    def clear_cache(self):
        """Clear fitness cache."""
        self.cache.clear()


# =========================
# GA Operators
# =========================
class GAOperators:
    """Genetic algorithm operators for graphs."""

    def __init__(self, config: GAConfig, validator: GraphValidator, rng: random.Random):
        self.config = config
        self.validator = validator
        self.rng = rng

    def initialize_population(
        self, base_graph: Dict[str, Any], population_size: int
    ) -> List[Dict[str, Any]]:
        """
        Initialize population by perturbing base graph.

        Args:
            base_graph: Base graph to perturb
            population_size: Number of graphs to generate

        Returns:
            List of valid graphs
        """
        population = []
        attempts = 0
        max_attempts = population_size * 10

        while len(population) < population_size and attempts < max_attempts:
            attempts += 1
            graph = copy.deepcopy(base_graph)

            # Generate unique ID
            unique_id = (
                f"{base_graph['id']}_gen0_ind{len(population)}_{uuid.uuid4().hex[:8]}"
            )
            graph["id"] = unique_id

            # Apply random perturbations
            for node in graph.get("nodes", []):
                if self.rng.random() < 0.3:  # 30% chance to perturb each node
                    self._perturb_node(node)

            # Validate
            is_valid, errors = self.validator.validate_graph(graph)
            if is_valid:
                population.append(graph)
            else:
                log.debug(f"Invalid initial graph: {errors}")

        if len(population) < population_size:
            log.warning(
                f"Only generated {len(population)}/{population_size} valid graphs"
            )

        log.info(f"Initialized population with {len(population)} graphs")
        return population

    def _perturb_node(self, node: Dict[str, Any]):
        """Apply small perturbation to a node."""
        if node.get("type") == "GenerativeNode" and "prompt" in node:
            variations = ["enhanced", "optimized", "refined", "improved"]
            node["prompt"] += f" [{self.rng.choice(variations)}]"
        elif "temperature" in node:
            # Adjust temperature slightly
            current = node.get("temperature", 0.7)
            node["temperature"] = max(0.0, min(1.0, current + self.rng.gauss(0, 0.1)))

    def mutate_graph(self, graph: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Mutate a graph by modifying nodes or edges.
        Validates result before returning.

        Args:
            graph: Graph to mutate

        Returns:
            Mutated graph if valid, None otherwise
        """
        mutated = copy.deepcopy(graph)

        # Generate new unique ID
        mutated["id"] = f"{graph['id']}_mut_{uuid.uuid4().hex[:8]}"

        nodes = mutated.get("nodes", [])
        if not nodes:
            return None

        # Randomly modify 1-3 nodes
        num_mutations = self.rng.randint(1, min(3, len(nodes)))
        for _ in range(num_mutations):
            node_idx = self.rng.randint(0, len(nodes) - 1)
            node = nodes[node_idx]
            self._perturb_node(node)

        # Validate
        is_valid, errors = self.validator.validate_graph(mutated)
        if is_valid:
            log.debug(f"Successfully mutated graph {graph['id']} -> {mutated['id']}")
            return mutated
        else:
            log.debug(f"Mutation produced invalid graph: {errors}")
            return None

    def crossover_graphs(
        self, parent1: Dict[str, Any], parent2: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Perform graph-aware crossover between two parents.
        Ensures edges remain valid after node exchange.

        Args:
            parent1: First parent graph
            parent2: Second parent graph

        Returns:
            Child graph if valid, None otherwise
        """
        child = copy.deepcopy(parent1)
        child["id"] = f"cross_{uuid.uuid4().hex[:8]}"

        nodes1 = parent1.get("nodes", [])
        nodes2 = parent2.get("nodes", [])
        edges1 = parent1.get("edges", [])

        if not nodes1 or not nodes2:
            return None

        # Select crossover point
        min_len = min(len(nodes1), len(nodes2))
        if min_len < 2:
            return None

        crossover_point = self.rng.randint(1, min_len - 1)

        # Swap nodes
        child_nodes = nodes1[:crossover_point] + nodes2[crossover_point:]
        child["nodes"] = child_nodes

        # Rebuild edges to maintain validity
        child_node_ids = {n["id"] for n in child_nodes}
        valid_edges = []

        for edge in edges1:
            # --- START OF THE FIX ---
            # Handle both simple string and complex dictionary references for 'from' and 'to'
            from_ref = edge["from"]
            to_ref = edge["to"]

            source_id = from_ref["node"] if isinstance(from_ref, dict) else from_ref
            target_id = to_ref["node"] if isinstance(to_ref, dict) else to_ref

            if source_id in child_node_ids and target_id in child_node_ids:
                valid_edges.append(copy.deepcopy(edge))
            # --- END OF THE FIX ---

        child["edges"] = valid_edges

        # Validate
        is_valid, errors = self.validator.validate_graph(child)
        if is_valid:
            log.debug(
                f"Successfully crossed {parent1['id']} and {parent2['id']} -> {child['id']}"
            )
            return child
        else:
            log.debug(f"Crossover produced invalid graph: {errors}")
            return None

    def tournament_selection(
        self, population: List[GraphCandidate], tournament_size: int
    ) -> GraphCandidate:
        """
        Perform tournament selection.

        Args:
            population: List of graph candidates
            tournament_size: Number of candidates in tournament

        Returns:
            Selected graph candidate
        """
        if len(population) < tournament_size:
            tournament_size = len(population)

        # Sample without replacement
        candidates = self.rng.sample(population, tournament_size)

        # Select best
        best = max(
            candidates, key=lambda c: c.metrics.score if c.metrics else float("-inf")
        )

        log.debug(
            f"Tournament selected {best.graph['id']} with score {best.metrics.score:.4f}"
        )
        return best


# =========================
# Checkpointing
# =========================
class CheckpointManager:
    """Manages checkpoints for resuming experiments."""

    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        generation: int,
        population: List[GraphCandidate],
        champions: List[GraphCandidate],
        rng_state: Any,
        config: GAConfig,
    ):
        """Save checkpoint to disk."""
        checkpoint = {
            "generation": generation,
            "population": [
                {
                    "graph": c.graph,
                    "metrics": (
                        {
                            "accuracy": c.metrics.accuracy,
                            "tokens": c.metrics.tokens,
                            "cycles": c.metrics.cycles,
                            "score": c.metrics.score,
                        }
                        if c.metrics
                        else None
                    ),
                    "generation": c.generation,
                    "parent_ids": c.parent_ids,
                }
                for c in population
            ],
            "champions": [
                {
                    "graph": c.graph,
                    "metrics": (
                        {
                            "accuracy": c.metrics.accuracy,
                            "tokens": c.metrics.tokens,
                            "cycles": c.metrics.cycles,
                            "score": c.metrics.score,
                        }
                        if c.metrics
                        else None
                    ),
                    "generation": c.generation,
                    "parent_ids": c.parent_ids,
                }
                for c in champions
            ],
            "rng_state": rng_state,
            "config": {
                "base_tokens": config.base_tokens,
                "tokens_per_node": config.tokens_per_node,
                "base_cycles": config.base_cycles,
                "cycles_per_node": config.cycles_per_node,
                "base_accuracy": config.base_accuracy,
                "accuracy_drop_per_node": config.accuracy_drop_per_node,
                "alpha_tokens": config.alpha_tokens,
                "beta_cycles_per_k": config.beta_cycles_per_k,
                "mutation_rate": config.mutation_rate,
                "crossover_rate": config.crossover_rate,
            },
        }

        # Write to temporary file first, then rename for atomicity
        checkpoint_file = self.checkpoint_dir / f"checkpoint_gen{generation}.json"
        temp_file = checkpoint_file.with_suffix(".tmp")

        try:
            with open(temp_file, "w") as f:
                json.dump(checkpoint, f, indent=2)
            temp_file.replace(checkpoint_file)
            log.info(
                f"Saved checkpoint at generation {generation} to {checkpoint_file}"
            )
        except Exception as e:
            log.error(f"Failed to save checkpoint: {e}")
            if temp_file.exists():
                temp_file.unlink()

    def load_checkpoint(self, checkpoint_file: Path) -> Optional[Dict[str, Any]]:
        """Load checkpoint from disk."""
        try:
            with open(checkpoint_file, "r") as f:
                checkpoint = json.load(f)
            log.info(f"Loaded checkpoint from {checkpoint_file}")
            return checkpoint
        except Exception as e:
            log.error(f"Failed to load checkpoint: {e}")
            return None


# =========================
# Offline Mode
# =========================
async def run_offline(args: argparse.Namespace, config: GAConfig) -> None:
    """
    Run tournament in offline mode using LanguageEvolutionRegistry.

    Args:
        args: CLI arguments
        config: GA configuration
    """
    if not HAS_REGISTRY:
        log.error("LanguageEvolutionRegistry not available for offline mode")
        raise RuntimeError("Offline mode requires LanguageEvolutionRegistry")

    # Initialize components
    backend = InMemoryBackend()
    kms = DevelopmentKMS()

    # Increase the rate limit specifically for this offline simulation.
    from specs.formal_grammar.language_evolution_registry import RateLimiter

    high_limit_rate_limiter = RateLimiter(max_per_hour=1000)

    registry = LanguageEvolutionRegistry(
        backend=backend, kms=kms, rate_limiter=high_limit_rate_limiter
    )
    log.info(
        "Initialized LanguageEvolutionRegistry for offline mode with high rate limit"
    )

    validator = GraphValidator(config)
    evaluator = FitnessEvaluator(config)
    operators = GAOperators(config, validator, args.rng)
    checkpoint_mgr = CheckpointManager(args.checkpoint_dir)

    # Validate and load files
    path_validator = PathValidator()

    is_valid, error = path_validator.validate_path(
        args.graph_file, args.graphs_dir.parent
    )
    if not is_valid:
        raise RuntimeError(f"Invalid graph file: {error}")

    is_valid, error = path_validator.validate_path(
        args.workload_file, args.graphs_dir.parent
    )
    if not is_valid:
        raise RuntimeError(f"Invalid workload file: {error}")

    try:
        with open(args.graph_file, "r") as f:
            base_graph = json.load(f)
        with open(args.workload_file, "r") as f:
            workload = json.load(f)
    except Exception as e:
        log.error(f"Failed to load files: {e}")
        raise RuntimeError(f"Failed to load files: {e}")

    # Validate base graph
    is_valid, errors = validator.validate_graph(base_graph)
    if not is_valid:
        raise RuntimeError(f"Base graph invalid: {errors}")

    # Initialize or resume
    start_generation = 0
    if args.resume and args.resume.exists():
        checkpoint = checkpoint_mgr.load_checkpoint(args.resume)
        if checkpoint:
            start_generation = checkpoint["generation"] + 1
            args.rng.setstate(checkpoint["rng_state"])
            # Reconstruct population
            population_candidates = [
                GraphCandidate(
                    graph=c["graph"],
                    metrics=Metrics(**c["metrics"]) if c["metrics"] else None,
                    generation=c["generation"],
                    parent_ids=c["parent_ids"],
                )
                for c in checkpoint["population"]
            ]
            champions = [
                GraphCandidate(
                    graph=c["graph"],
                    metrics=Metrics(**c["metrics"]) if c["metrics"] else None,
                    generation=c["generation"],
                    parent_ids=c["parent_ids"],
                )
                for c in checkpoint["champions"]
            ]
            log.info(f"Resumed from generation {start_generation}")
        else:
            population_candidates = []
            champions = []
    else:
        # Initialize population
        population_graphs = operators.initialize_population(base_graph, args.population)
        population_candidates = [
            GraphCandidate(graph=g, generation=0) for g in population_graphs
        ]
        champions = []

    # Evaluate initial population
    for candidate in population_candidates:
        if candidate.metrics is None:
            candidate.metrics = evaluator.evaluate(candidate.graph)

    # Track best fitness for early stopping
    best_fitness_history = []

    # Run GA
    for generation in range(start_generation, args.generations):
        log.info(f"Starting generation {generation + 1}/{args.generations}")

        # Submit proposals to registry for each candidate
        for candidate in population_candidates:
            try:
                proposal = {
                    "type": "ProposalNode",
                    "proposed_by": "tournament-agent-offline",
                    "rationale": f"GA candidate from generation {generation}",
                    "proposal_content": {
                        "add": {candidate.graph["id"]: candidate.graph}
                    },
                    "metadata": {
                        "author": "tournament-agent-offline",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "version": "1.0.0",
                        "generation": generation,
                        "fitness_score": (
                            candidate.metrics.score if candidate.metrics else 0.0
                        ),
                    },
                }

                proposal_id = registry.submit_proposal(proposal)
                log.debug(
                    f"Submitted proposal {proposal_id} for graph {candidate.graph['id']}"
                )

                # Simulate voting (auto-approve for now)
                consensus = {
                    "proposal_id": proposal_id,
                    "votes": {"tournament-agent-offline": "yes"},
                    "weights": {"tournament-agent-offline": 1.0},
                }
                registry.record_vote(consensus)

            except Exception as e:
                log.warning(
                    f"Failed to process graph {candidate.graph['id']} in registry: {e}"
                )

        # Sort by fitness
        population_candidates.sort(
            key=lambda c: c.metrics.score if c.metrics else float("-inf"), reverse=True
        )

        # Log statistics
        scores = [c.metrics.score for c in population_candidates if c.metrics]
        if scores:
            best_score = max(scores)
            avg_score = sum(scores) / len(scores)
            diversity = len(set(c.graph["id"] for c in population_candidates)) / len(
                population_candidates
            )

            log.info(
                f"Gen {generation}: Best={best_score:.4f}, Avg={avg_score:.4f}, "
                f"Diversity={diversity:.2f}, PopSize={len(population_candidates)}"
            )

            best_fitness_history.append(best_score)

        # Early stopping check
        if len(best_fitness_history) >= config.convergence_window:
            recent = best_fitness_history[-config.convergence_window :]
            if max(recent) - min(recent) < config.convergence_threshold:
                log.info(
                    f"Converged after {generation + 1} generations, stopping early"
                )
                break

        # Select elites
        elites = population_candidates[: args.elites]

        # Add to champions (limit size)
        champions.extend(elites)
        if len(champions) > config.max_champions:
            champions.sort(
                key=lambda c: c.metrics.score if c.metrics else float("-inf"),
                reverse=True,
            )
            champions = champions[: config.max_champions]

        log.info(f"Selected {len(elites)} elites, total champions: {len(champions)}")

        # Generate new population
        new_population = [
            GraphCandidate(
                graph=copy.deepcopy(e.graph),
                generation=generation + 1,
                parent_ids=[e.graph["id"]],
            )
            for e in elites
        ]

        attempts = 0
        max_attempts = args.population * 5

        while len(new_population) < args.population and attempts < max_attempts:
            attempts += 1

            # Select parents
            parent1 = operators.tournament_selection(
                population_candidates, args.tournament
            )
            parent2 = operators.tournament_selection(
                population_candidates, args.tournament
            )

            # Ensure parents are different
            retry_count = 0
            while parent1.graph["id"] == parent2.graph["id"] and retry_count < 10:
                parent2 = operators.tournament_selection(
                    population_candidates, args.tournament
                )
                retry_count += 1

            # Crossover or clone
            if args.rng.random() < config.crossover_rate:
                child_graph = operators.crossover_graphs(parent1.graph, parent2.graph)
                parent_ids = [parent1.graph["id"], parent2.graph["id"]]
            else:
                child_graph = copy.deepcopy(parent1.graph)
                child_graph["id"] = f"clone_{uuid.uuid4().hex[:8]}"
                parent_ids = [parent1.graph["id"]]

            # Mutation
            if child_graph and args.rng.random() < config.mutation_rate:
                mutated = operators.mutate_graph(child_graph)
                if mutated:
                    child_graph = mutated

            # Add to population if valid
            if child_graph:
                candidate = GraphCandidate(
                    graph=child_graph, generation=generation + 1, parent_ids=parent_ids
                )
                candidate.metrics = evaluator.evaluate(child_graph)
                new_population.append(candidate)

        if len(new_population) < args.population:
            log.warning(
                f"Only generated {len(new_population)}/{args.population} valid offspring"
            )

        population_candidates = new_population

        # Save checkpoint
        if (generation + 1) % args.checkpoint_interval == 0:
            checkpoint_mgr.save_checkpoint(
                generation=generation,
                population=population_candidates,
                champions=champions,
                rng_state=args.rng.getstate(),
                config=config,
            )

        # --- START OF THE FIX ---
        # Add a delay between generations to allow the rate limiter's token bucket to recover.
        if generation < args.generations - 1:  # Don't sleep after the last generation
            log.info(
                f"Pausing for 2 seconds before next generation to respect rate limits..."
            )
            await asyncio.sleep(2)
        # --- END OF THE FIX ---

    # Save final champion
    if champions:
        champion = max(
            champions, key=lambda c: c.metrics.score if c.metrics else float("-inf")
        )

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_dir = args.champions_dir / f"offline_{timestamp}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Write champion atomically
        champion_file = out_dir / "champion_final.json"
        temp_file = champion_file.with_suffix(".tmp")

        champion_data = {
            "graph": champion.graph,
            "metrics": (
                {
                    "accuracy": champion.metrics.accuracy,
                    "tokens": champion.metrics.tokens,
                    "cycles": champion.metrics.cycles,
                    "score": champion.metrics.score,
                }
                if champion.metrics
                else None
            ),
            "generation": champion.generation,
            "parent_ids": champion.parent_ids,
        }

        with open(temp_file, "w") as f:
            json.dump(champion_data, f, indent=2)
        temp_file.replace(champion_file)

        # Save workload
        workload_file = out_dir / "champion_eval_workload.json"
        temp_file = workload_file.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(workload, f, indent=2)
        temp_file.replace(workload_file)

        # Save metrics history
        metrics_file = out_dir / "metrics_history.json"
        temp_file = metrics_file.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump({"best_fitness": best_fitness_history}, f, indent=2)
        temp_file.replace(metrics_file)

        log.info(f"Champion saved to: {out_dir}")
        log.info(f"Final champion score: {champion.metrics.score:.4f}")

    log.info("Offline tournament complete.")


# =========================
# Online Mode
# =========================
async def run_online(
    args: argparse.Namespace, config: GAConfig, client: "AgentInterface"
) -> None:
    """
    Run tournament in online mode using AgentInterface.

    Args:
        args: CLI arguments
        config: GA configuration
        client: AgentInterface instance
    """
    if not HAS_SDK:
        log.error("AgentInterface not available for online mode")
        raise RuntimeError("Online mode requires AgentInterface")

    log.info("Running in online mode with AgentInterface")

    validator = GraphValidator(config)
    evaluator = FitnessEvaluator(config)
    operators = GAOperators(config, validator, args.rng)
    checkpoint_mgr = CheckpointManager(args.checkpoint_dir)

    # Validate and load files
    path_validator = PathValidator()

    is_valid, error = path_validator.validate_path(
        args.graph_file, args.graphs_dir.parent
    )
    if not is_valid:
        raise RuntimeError(f"Invalid graph file: {error}")

    is_valid, error = path_validator.validate_path(
        args.workload_file, args.graphs_dir.parent
    )
    if not is_valid:
        raise RuntimeError(f"Invalid workload file: {error}")

    try:
        with open(args.graph_file, "r") as f:
            base_graph = json.load(f)
        with open(args.workload_file, "r") as f:
            workload = json.load(f)
    except Exception as e:
        log.error(f"Failed to load files: {e}")
        raise RuntimeError(f"Failed to load files: {e}")

    # Validate base graph
    is_valid, errors = validator.validate_graph(base_graph)
    if not is_valid:
        raise RuntimeError(f"Base graph invalid: {errors}")

    # Initialize population
    population_graphs = operators.initialize_population(base_graph, args.population)
    population_candidates = [
        GraphCandidate(graph=g, generation=0) for g in population_graphs
    ]
    champions = []
    best_fitness_history = []

    # Run GA
    for generation in range(args.generations):
        log.info(f"Starting generation {generation + 1}/{args.generations}")

        # Evaluate population in parallel
        tasks = []
        for candidate in population_candidates:
            if candidate.metrics is None:
                tasks.append(
                    evaluate_candidate_online(client, candidate, evaluator, args)
                )

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    log.error(f"Evaluation failed: {result}")

        # Filter out failed evaluations
        population_candidates = [
            c for c in population_candidates if c.metrics is not None
        ]

        if not population_candidates:
            log.error("No valid candidates remaining, aborting")
            break

        # Sort by fitness
        population_candidates.sort(key=lambda c: c.metrics.score, reverse=True)

        # Log statistics
        scores = [c.metrics.score for c in population_candidates]
        best_score = max(scores)
        avg_score = sum(scores) / len(scores)

        log.info(
            f"Gen {generation}: Best={best_score:.4f}, Avg={avg_score:.4f}, "
            f"PopSize={len(population_candidates)}"
        )

        best_fitness_history.append(best_score)

        # Early stopping
        if len(best_fitness_history) >= config.convergence_window:
            recent = best_fitness_history[-config.convergence_window :]
            if max(recent) - min(recent) < config.convergence_threshold:
                log.info(f"Converged after {generation + 1} generations")
                break

        # Select elites
        elites = population_candidates[: args.elites]
        champions.extend(elites)

        if len(champions) > config.max_champions:
            champions.sort(key=lambda c: c.metrics.score, reverse=True)
            champions = champions[: config.max_champions]

        # Generate new population
        new_population = [
            GraphCandidate(
                graph=copy.deepcopy(e.graph),
                generation=generation + 1,
                parent_ids=[e.graph["id"]],
            )
            for e in elites
        ]

        attempts = 0
        max_attempts = args.population * 5

        while len(new_population) < args.population and attempts < max_attempts:
            attempts += 1

            parent1 = operators.tournament_selection(
                population_candidates, args.tournament
            )
            parent2 = operators.tournament_selection(
                population_candidates, args.tournament
            )

            # Ensure different parents
            retry = 0
            while parent1.graph["id"] == parent2.graph["id"] and retry < 10:
                parent2 = operators.tournament_selection(
                    population_candidates, args.tournament
                )
                retry += 1

            # Crossover or clone
            if args.rng.random() < config.crossover_rate:
                child_graph = operators.crossover_graphs(parent1.graph, parent2.graph)
                parent_ids = [parent1.graph["id"], parent2.graph["id"]]
            else:
                child_graph = copy.deepcopy(parent1.graph)
                child_graph["id"] = f"clone_{uuid.uuid4().hex[:8]}"
                parent_ids = [parent1.graph["id"]]

            # Mutation
            if child_graph and args.rng.random() < config.mutation_rate:
                mutated = operators.mutate_graph(child_graph)
                if mutated:
                    child_graph = mutated

            if child_graph:
                new_population.append(
                    GraphCandidate(
                        graph=child_graph,
                        generation=generation + 1,
                        parent_ids=parent_ids,
                    )
                )

        population_candidates = new_population

        # Checkpoint
        if (generation + 1) % args.checkpoint_interval == 0:
            checkpoint_mgr.save_checkpoint(
                generation=generation,
                population=population_candidates,
                champions=champions,
                rng_state=args.rng.getstate(),
                config=config,
            )

    # Save champion
    if champions:
        champion = max(champions, key=lambda c: c.metrics.score)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_dir = args.champions_dir / f"online_{timestamp}"
        out_dir.mkdir(parents=True, exist_ok=True)

        champion_data = {
            "graph": champion.graph,
            "metrics": {
                "accuracy": champion.metrics.accuracy,
                "tokens": champion.metrics.tokens,
                "cycles": champion.metrics.cycles,
                "score": champion.metrics.score,
            },
            "generation": champion.generation,
            "parent_ids": champion.parent_ids,
        }

        # Atomic write
        champion_file = out_dir / "champion_final.json"
        temp_file = champion_file.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(champion_data, f, indent=2)
        temp_file.replace(champion_file)

        # Save workload
        workload_file = out_dir / "champion_eval_workload.json"
        temp_file = workload_file.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(workload, f, indent=2)
        temp_file.replace(workload_file)

        log.info(f"Champion saved to: {out_dir}")
        log.info(f"Final champion score: {champion.metrics.score:.4f}")

    log.info("Online tournament complete.")


async def evaluate_candidate_online(
    client: "AgentInterface",
    candidate: GraphCandidate,
    evaluator: FitnessEvaluator,
    args: argparse.Namespace,
) -> GraphCandidate:
    """Evaluate a candidate using online services."""
    try:
        # FIXED: Use submit_graph method from AgentInterface
        execution_result = await client.submit_graph(
            ir_graph=candidate.graph, wait_for_result=True, timeout=args.timeout
        )

        # Extract metrics
        if execution_result and "metrics" in execution_result:
            metrics_data = execution_result["metrics"]
            candidate.metrics = Metrics(
                accuracy=metrics_data.get("accuracy", 0.0),
                tokens=metrics_data.get("tokens", 0.0),
                cycles=metrics_data.get("cycles", 0.0),
                score=metrics_data.get("score", 0.0),
            )
        else:
            # Fallback to heuristic
            log.warning(
                f"Online execution for {candidate.graph['id']} failed or returned no metrics. Falling back."
            )
            candidate.metrics = evaluator.evaluate(candidate.graph)

        return candidate
    except Exception as e:
        log.error(f"Error evaluating {candidate.graph['id']}: {e}")
        # Fallback to heuristic
        candidate.metrics = evaluator.evaluate(candidate.graph)
        return candidate


# =========================
# CLI
# =========================
def main() -> None:
    p = argparse.ArgumentParser(
        description="Graphix GA/Tournament driver",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode
    p.add_argument(
        "--mode",
        choices=["offline", "online"],
        default="offline",
        help="Run mode: offline (registry simulation) or online (live services)",
    )

    # GA parameters
    p.add_argument(
        "--generations", type=int, default=6, help="Number of generations to run"
    )
    p.add_argument(
        "--population", type=int, default=8, help="Population size per generation"
    )
    p.add_argument(
        "--tournament",
        type=int,
        default=3,
        help="Tournament size for selection (2-5 recommended)",
    )
    p.add_argument(
        "--elites",
        type=int,
        default=1,
        help="Number of elites to preserve each generation",
    )

    # File paths
    p.add_argument(
        "--graph-file",
        type=Path,
        default=DEFAULT_SENTIMENT_GRAPH,
        help="Path to base graph JSON file",
    )
    p.add_argument(
        "--workload-file",
        type=Path,
        default=DEFAULT_EVAL_WORKLOAD,
        help="Path to evaluation workload JSON file",
    )
    p.add_argument(
        "--graphs-dir",
        type=Path,
        default=DEFAULT_GRAPHS_DIR,
        help="Base directory for graphs (for path validation)",
    )
    p.add_argument(
        "--champions-dir",
        type=Path,
        default=DEFAULT_CHAMPIONS_OUT,
        help="Output directory for champion artifacts",
    )
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory for checkpoints",
    )
    p.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5,
        help="Save checkpoint every N generations",
    )
    p.add_argument(
        "--resume", type=Path, default=None, help="Resume from checkpoint file"
    )

    # Online mode parameters
    p.add_argument(
        "--host",
        default="http://localhost:8787",
        help="Host endpoint for online mode (e.g., http://localhost:8787)",
    )
    p.add_argument(
        "--api-key", type=str, default=None, help="API key for online mode (required)"
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Timeout in seconds for online graph execution",
    )

    # GA configuration
    p.add_argument(
        "--mutation-rate",
        type=float,
        default=0.1,
        help="Mutation probability (0.0-1.0)",
    )
    p.add_argument(
        "--crossover-rate",
        type=float,
        default=0.8,
        help="Crossover probability (0.0-1.0)",
    )

    # Other
    p.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity (-v, -vv)",
    )

    args = p.parse_args()

    # Logging level
    if args.verbose >= 2:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose == 1:
        logging.getLogger().setLevel(logging.INFO)

    # Initialize RNG (always create instance for consistency)
    rng = random.Random(args.seed)
    args.rng = rng

    # Create GA config
    config = GAConfig(
        mutation_rate=args.mutation_rate, crossover_rate=args.crossover_rate
    )

    # Run
    try:
        if args.mode == "offline":
            asyncio.run(run_offline(args, config))
        else:
            if not HAS_SDK:
                log.error("Graphix client is not available. Cannot run in online mode.")
                sys.exit(1)
            if not args.api_key:
                log.error("API key is required for online mode.")
                sys.exit(1)

            # FIXED: Instantiate AgentInterface and use it as a context manager
            # The host URL needs the http:// prefix
            host_url = (
                args.host if args.host.startswith("http") else f"http://{args.host}"
            )
            # Extract just the hostname for the ConnectionConfig
            parsed_url = urlparse(host_url)

            conn_config = ConnectionConfig(
                host=parsed_url.hostname,
                port=parsed_url.port or 80,
                api_key=args.api_key,
            )

            # The interface must be used as a context manager to connect/disconnect
            async def online_main():
                async with AgentInterface(config=conn_config) as interface:
                    await run_online(args, config, interface)

            asyncio.run(online_main())

    except KeyboardInterrupt:
        log.info("Interrupted by user")
    except Exception as e:
        log.error(f"Tournament failed with an unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
