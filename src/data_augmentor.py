"""
Graphix Data Augmentor (Production-Ready)
==========================================
Version: 2.0.0 - All crashes fixed, semantic augmentation implemented
Generates high-quality synthetic, counterfactual, and adversarial graph proposals.
"""

import copy
import hashlib
import json
import logging
import random
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("DataAugmentor")

# Constants
MAX_COMPLEXITY = 10
MAX_NODES = 1000
MAX_EDGES = 5000
MAX_BATCH_SIZE = 100
MIN_DIVERSITY_THRESHOLD = 0.3
QUALITY_SCORE_WEIGHTS = {
    "connectivity": 0.3,
    "diversity": 0.3,
    "semantic_validity": 0.4,
}


@dataclass
class AugmentationMetrics:
    """Track augmentation quality metrics."""

    total_generated: int = 0
    synthetic_count: int = 0
    counterfactual_count: int = 0
    adversarial_count: int = 0
    duplicates_detected: int = 0
    invalid_graphs: int = 0
    avg_quality_score: float = 0.0
    diversity_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_generated": self.total_generated,
            "synthetic_count": self.synthetic_count,
            "counterfactual_count": self.counterfactual_count,
            "adversarial_count": self.adversarial_count,
            "duplicates_detected": self.duplicates_detected,
            "invalid_graphs": self.invalid_graphs,
            "avg_quality_score": self.avg_quality_score,
            "diversity_score": self.diversity_score,
        }


class GraphValidator:
    """Validates graph structure and semantics."""

    @staticmethod
    def validate_graph(graph: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate graph structure.

        Returns:
            (is_valid, error_message) tuple
        """
        if not isinstance(graph, dict):
            return False, "Graph must be a dictionary"

        # Check required fields
        if "nodes" not in graph:
            return False, "Missing 'nodes' field"

        nodes = graph.get("nodes", [])
        if not isinstance(nodes, list):
            return False, "'nodes' must be a list"

        if len(nodes) > MAX_NODES:
            return False, f"Too many nodes: {len(nodes)} > {MAX_NODES}"

        # Validate nodes
        node_ids = set()
        for i, node in enumerate(nodes):
            if not isinstance(node, dict):
                return False, f"Node {i} is not a dictionary"

            if "id" not in node:
                return False, f"Node {i} missing 'id' field"

            node_id = node["id"]
            if node_id in node_ids:
                return False, f"Duplicate node id: {node_id}"

            node_ids.add(node_id)

        # Validate edges if present
        edges = graph.get("edges", [])
        if not isinstance(edges, list):
            return False, "'edges' must be a list"

        if len(edges) > MAX_EDGES:
            return False, f"Too many edges: {len(edges)} > {MAX_EDGES}"

        for i, edge in enumerate(edges):
            if not isinstance(edge, dict):
                return False, f"Edge {i} is not a dictionary"

            # Check edge connectivity
            source = edge.get("from")
            target = edge.get("to")

            if source and source not in node_ids:
                return False, f"Edge {i} references non-existent source: {source}"

            if target and target not in node_ids:
                return False, f"Edge {i} references non-existent target: {target}"

        return True, None

    @staticmethod
    def validate_node(node: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate individual node structure."""
        if not isinstance(node, dict):
            return False, "Node must be a dictionary"

        if "id" not in node:
            return False, "Node missing 'id' field"

        if not isinstance(node["id"], str):
            return False, "Node 'id' must be a string"

        return True, None

    @staticmethod
    def validate_edge(
        edge: Dict[str, Any], node_ids: Set[str]
    ) -> Tuple[bool, Optional[str]]:
        """Validate individual edge structure."""
        if not isinstance(edge, dict):
            return False, "Edge must be a dictionary"

        source = edge.get("from")
        target = edge.get("to")

        if source and source not in node_ids:
            return False, f"Edge references non-existent source: {source}"

        if target and target not in node_ids:
            return False, f"Edge references non-existent target: {target}"

        return True, None


class SemanticMutator:
    """Semantic-aware graph mutations."""

    # Node type templates for semantic augmentation
    NODE_TEMPLATES = {
        "Sentiment": ["Emotion", "Opinion", "Attitude", "Feeling"],
        "Score": ["Rating", "Value", "Metric", "Measure"],
        "Data": ["Information", "Content", "Input", "Stream"],
        "Process": ["Transform", "Operation", "Function", "Handler"],
        "Output": ["Result", "Response", "Yield", "Product"],
    }

    # Semantic relationships
    SEMANTIC_RELATIONS = {
        "depends_on": {"weight_range": (0.5, 1.0), "inverse": "supports"},
        "supports": {"weight_range": (0.3, 0.8), "inverse": "depends_on"},
        "conflicts_with": {"weight_range": (-1.0, -0.3), "inverse": "conflicts_with"},
        "similar_to": {"weight_range": (0.6, 0.9), "inverse": "similar_to"},
        "transforms_to": {"weight_range": (0.4, 0.9), "inverse": "derived_from"},
    }

    @classmethod
    def mutate_node_semantic(
        cls, node: Dict[str, Any], rng: random.Random
    ) -> Dict[str, Any]:
        """
        Semantically meaningful node mutation.

        Args:
            node: Node to mutate
            rng: Random number generator

        Returns:
            Mutated node
        """
        mutated = copy.deepcopy(node)
        label = mutated.get("label", "Node")

        # Try to find semantic replacement
        for base_type, alternatives in cls.NODE_TEMPLATES.items():
            if base_type in label or label in alternatives:
                new_label = rng.choice(alternatives)
                mutated["label"] = new_label
                break
        else:
            # Generic mutation
            mutation_ops = ["prefix", "suffix", "variant"]
            op = rng.choice(mutation_ops)

            if op == "prefix":
                mutated["label"] = f"Enhanced{label}"
            elif op == "suffix":
                mutated["label"] = f"{label}V2"
            else:
                mutated["label"] = f"{label}Variant"

        # Add semantic metadata
        if "properties" not in mutated:
            mutated["properties"] = {}

        mutated["properties"]["mutation_type"] = "semantic"
        mutated["properties"]["original_label"] = label

        return mutated

    @classmethod
    def create_semantic_edge(
        cls,
        from_id: str,
        to_id: str,
        rng: random.Random,
        relation_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create semantically meaningful edge.

        Args:
            from_id: Source node ID
            to_id: Target node ID
            rng: Random number generator
            relation_type: Optional specific relation type

        Returns:
            Edge dictionary
        """
        if relation_type and relation_type in cls.SEMANTIC_RELATIONS:
            relation = relation_type
        else:
            relation = rng.choice(list(cls.SEMANTIC_RELATIONS.keys()))

        config = cls.SEMANTIC_RELATIONS[relation]
        weight = rng.uniform(*config["weight_range"])

        return {
            "from": from_id,
            "to": to_id,
            "weight": round(weight, 3),
            "relation_type": relation,
            "properties": {"semantic": True, "inverse_relation": config["inverse"]},
        }


class DataAugmentor:
    """
    Production-ready graph augmentation with semantic understanding.

    Features:
    - Thread-safe operations
    - Comprehensive validation
    - Semantic-aware mutations
    - Quality scoring
    - Diversity tracking
    - Duplicate detection
    - Audit logging
    """

    def __init__(
        self,
        random_seed: Optional[int] = None,
        audit_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """
        Initialize data augmentor.

        Args:
            random_seed: Random seed for reproducibility
            audit_hook: Optional callback for audit events
        """
        # Thread-safe random number generator
        self.rng = random.Random(random_seed)

        # Thread safety
        self.lock = threading.RLock()
        self.counter = 0

        # Audit and metrics
        self.audit_hook = audit_hook
        self.audit_log: List[Dict[str, Any]] = []
        self.metrics = AugmentationMetrics()

        # Diversity tracking
        self.generated_hashes: Set[str] = set()
        self.generated_graphs: List[Dict[str, Any]] = []

        # Validators
        self.validator = GraphValidator()
        self.mutator = SemanticMutator()

        logger.info("DataAugmentor initialized")

    def _get_counter(self) -> int:
        """Thread-safe counter increment."""
        with self.lock:
            self.counter += 1
            return self.counter

    def _hash_proposal(self, proposal: Dict[str, Any], kind: str) -> str:
        """
        Generate deterministic hash for proposal.

        Args:
            proposal: Proposal to hash
            kind: Augmentation kind

        Returns:
            Hash string
        """

        # Sort keys recursively for deterministic JSON
        def sort_dict(d):
            if isinstance(d, dict):
                return {k: sort_dict(v) for k, v in sorted(d.items())}
            elif isinstance(d, list):
                return [sort_dict(x) for x in d]
            return d

        sorted_proposal = sort_dict(proposal)
        content = json.dumps(sorted_proposal, sort_keys=True) + f":{kind}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _check_duplicate(self, proposal_hash: str) -> bool:
        """Check if proposal is duplicate."""
        with self.lock:
            if proposal_hash in self.generated_hashes:
                self.metrics.duplicates_detected += 1
                return True
            self.generated_hashes.add(proposal_hash)
            return False

    def _calculate_quality_score(self, proposal: Dict[str, Any]) -> float:
        """
        Calculate quality score for augmented proposal.

        Returns:
            Quality score (0-1)
        """
        score = 0.0

        nodes = proposal.get("nodes", [])
        edges = proposal.get("edges", [])

        # Connectivity score
        if nodes:
            connectivity = len(edges) / max(len(nodes), 1)
            connectivity_score = min(connectivity / 2.0, 1.0)
            score += QUALITY_SCORE_WEIGHTS["connectivity"] * connectivity_score

        # Diversity score (from original)
        augmented_count = sum(
            1 for n in nodes if n.get("properties", {}).get("augmented")
        )
        if nodes:
            diversity = augmented_count / len(nodes)
            score += QUALITY_SCORE_WEIGHTS["diversity"] * diversity

        # Semantic validity (has proper structure)
        has_ids = all("id" in n for n in nodes)
        has_labels = all("label" in n for n in nodes)
        semantic_score = (int(has_ids) + int(has_labels)) / 2.0
        score += QUALITY_SCORE_WEIGHTS["semantic_validity"] * semantic_score

        return min(max(score, 0.0), 1.0)

    def _update_metrics(self, kind: str, quality: float):
        """Update augmentation metrics."""
        with self.lock:
            self.metrics.total_generated += 1

            if kind == "synthetic":
                self.metrics.synthetic_count += 1
            elif kind == "counterfactual":
                self.metrics.counterfactual_count += 1
            elif kind == "adversarial":
                self.metrics.adversarial_count += 1

            # Update rolling average
            n = self.metrics.total_generated
            self.metrics.avg_quality_score = (
                self.metrics.avg_quality_score * (n - 1) + quality
            ) / n

    def _audit(self, kind: str, proposal: Dict[str, Any], quality: float = 0.0):
        """
        Comprehensive audit logging.

        Args:
            kind: Augmentation type
            proposal: Generated proposal
            quality: Quality score
        """
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "kind": kind,
            "proposal_id": proposal.get("metadata", {}).get(f"{kind[:3]}_id"),
            "quality_score": quality,
            "node_count": len(proposal.get("nodes", [])),
            "edge_count": len(proposal.get("edges", [])),
            "is_valid": self.validator.validate_graph(proposal)[0],
        }

        with self.lock:
            self.audit_log.append(audit_entry)

        # Call external hook if provided
        if self.audit_hook:
            try:
                self.audit_hook(audit_entry)
            except Exception as e:
                logger.error(f"Audit hook failed: {e}")

        # Log to file
        logger.debug(f"Audit: {kind} - quality={quality:.3f}")

    def generate_synthetic_proposal(
        self, base_graph: Dict[str, Any], complexity: int = 1
    ) -> Dict[str, Any]:
        """
        Generate synthetic proposal with semantic mutations.

        Args:
            base_graph: Base graph to augment
            complexity: Number of mutations (1-10)

        Returns:
            Augmented proposal

        Raises:
            ValueError: If base graph invalid or complexity out of range
        """
        # Validate input
        valid, error = self.validator.validate_graph(base_graph)
        if not valid:
            raise ValueError(f"Invalid base graph: {error}")

        if not (1 <= complexity <= MAX_COMPLEXITY):
            raise ValueError(f"Complexity must be 1-{MAX_COMPLEXITY}, got {complexity}")

        # Deep copy to avoid modifying original
        proposal = copy.deepcopy(base_graph)
        nodes = proposal.get("nodes", [])
        edges = proposal.get("edges", [])

        # Get thread-safe counter
        counter = self._get_counter()

        # Perform mutations
        for mutation_idx in range(complexity):
            # Choose operation based on current state
            operations = []

            if nodes:
                operations.extend(
                    ["mutate_node", "mutate_edge" if edges else "add_edge"]
                )

            operations.append("add_node")

            if len(nodes) > 1:
                operations.append("add_edge")

            if not operations:
                operations = ["add_node"]

            op = self.rng.choice(operations)

            if op == "mutate_node" and nodes:
                # Semantic node mutation
                node_idx = self.rng.randint(0, len(nodes) - 1)
                nodes[node_idx] = self.mutator.mutate_node_semantic(
                    nodes[node_idx], self.rng
                )
                nodes[node_idx]["properties"]["augmented"] = True
                nodes[node_idx]["properties"]["mutation_index"] = mutation_idx

            elif op == "mutate_edge" and edges:
                # Mutate edge weight and properties
                edge_idx = self.rng.randint(0, len(edges) - 1)
                edges[edge_idx]["weight"] = round(self.rng.uniform(0.1, 5.0), 3)

                if "properties" not in edges[edge_idx]:
                    edges[edge_idx]["properties"] = {}

                edges[edge_idx]["properties"]["augmented"] = True
                edges[edge_idx]["properties"]["mutation_index"] = mutation_idx

            elif op == "add_node":
                # Add semantic node
                new_node_id = (
                    f"syn_node_{self.rng.randint(1000, 9999)}_{counter}_{mutation_idx}"
                )
                node_type = self.rng.choice(["Data", "Process", "Output", "Transform"])

                nodes.append(
                    {
                        "id": new_node_id,
                        "label": f"{node_type}_Synthetic",
                        "properties": {
                            "augmented": True,
                            "synthetic": True,
                            "mutation_index": mutation_idx,
                        },
                    }
                )

            elif op == "add_edge" and len(nodes) > 1:
                # Add semantic edge
                available_nodes = [n for n in nodes if "id" in n]

                if len(available_nodes) >= 2:
                    src_node = self.rng.choice(available_nodes)
                    src_id = src_node["id"]

                    # Choose different target
                    target_candidates = [
                        n for n in available_nodes if n["id"] != src_id
                    ]

                    if target_candidates:
                        dst_node = self.rng.choice(target_candidates)
                        dst_id = dst_node["id"]

                        # Create semantic edge
                        new_edge = self.mutator.create_semantic_edge(
                            src_id, dst_id, self.rng
                        )
                        new_edge["properties"]["augmented"] = True
                        new_edge["properties"]["mutation_index"] = mutation_idx

                        edges.append(new_edge)

        # Update proposal metadata
        if "metadata" not in proposal:
            proposal["metadata"] = {}

        proposal["metadata"]["synthetic"] = True
        proposal["metadata"]["complexity"] = complexity
        proposal["metadata"]["mutations"] = complexity
        proposal["metadata"]["aug_id"] = self._hash_proposal(proposal, "synthetic")
        proposal["metadata"]["generated_at"] = datetime.utcnow().isoformat()

        # Calculate quality and check duplicate
        quality = self._calculate_quality_score(proposal)
        proposal["metadata"]["quality_score"] = quality

        proposal_hash = self._hash_proposal(proposal, "synthetic")
        is_duplicate = self._check_duplicate(proposal_hash)

        if is_duplicate:
            logger.warning(f"Duplicate synthetic proposal generated: {proposal_hash}")

        # Update metrics and audit
        self._update_metrics("synthetic", quality)
        self._audit("synthetic", proposal, quality)

        with self.lock:
            self.generated_graphs.append(proposal)

        return proposal

    def counterfactual_proposal(
        self, base_graph: Dict[str, Any], invert_all: bool = False
    ) -> Dict[str, Any]:
        """
        Generate counterfactual proposal with semantic negation.

        Args:
            base_graph: Base graph to invert
            invert_all: If True, invert all nodes/edges; else just first

        Returns:
            Counterfactual proposal

        Raises:
            ValueError: If base graph invalid
        """
        # Validate input
        valid, error = self.validator.validate_graph(base_graph)
        if not valid:
            raise ValueError(f"Invalid base graph: {error}")

        proposal = copy.deepcopy(base_graph)
        nodes = proposal.get("nodes", [])
        edges = proposal.get("edges", [])
        self._get_counter()

        # Semantic negation of nodes
        if nodes:
            for i, node in enumerate(nodes):
                if i == 0 or invert_all:
                    label = node.get("label", "Node")

                    # Semantic negation strategies
                    negation_strategies = {
                        "NOT_": lambda l: (
                            l.replace("NOT_", "") if "NOT_" in l else f"NOT_{l}"
                        ),
                        "Anti": lambda l: (
                            l.replace("Anti", "")
                            if l.startswith("Anti")
                            else f"Anti{l}"
                        ),
                        "Inverse": lambda l: (
                            l.replace("Inverse", "")
                            if l.startswith("Inverse")
                            else f"Inverse{l}"
                        ),
                    }

                    # Choose strategy
                    strategy = self.rng.choice(list(negation_strategies.keys()))
                    node["label"] = negation_strategies[strategy](label)

                    if "properties" not in node:
                        node["properties"] = {}

                    node["properties"]["counterfactual"] = True
                    node["properties"]["original_label"] = label
                    node["properties"]["negation_strategy"] = strategy

        # Invert edge weights
        for i, edge in enumerate(edges):
            if i == 0 or invert_all:
                if "weight" in edge:
                    original_weight = edge["weight"]
                    edge["weight"] = -original_weight

                    if "properties" not in edge:
                        edge["properties"] = {}

                    edge["properties"]["counterfactual"] = True
                    edge["properties"]["original_weight"] = original_weight

        # Update metadata
        if "metadata" not in proposal:
            proposal["metadata"] = {}

        proposal["metadata"]["counterfactual"] = True
        proposal["metadata"]["invert_all"] = invert_all
        proposal["metadata"]["cf_id"] = self._hash_proposal(proposal, "counterfactual")
        proposal["metadata"]["generated_at"] = datetime.utcnow().isoformat()

        # Calculate quality
        quality = self._calculate_quality_score(proposal)
        proposal["metadata"]["quality_score"] = quality

        # Check duplicate
        proposal_hash = self._hash_proposal(proposal, "counterfactual")
        is_duplicate = self._check_duplicate(proposal_hash)

        if is_duplicate:
            logger.warning(f"Duplicate counterfactual proposal: {proposal_hash}")

        # Update metrics and audit
        self._update_metrics("counterfactual", quality)
        self._audit("counterfactual", proposal, quality)

        with self.lock:
            self.generated_graphs.append(proposal)

        return proposal

    def adversarial_proposal(
        self, base_graph: Dict[str, Any], targeted_node: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate adversarial proposal for robustness testing.

        Args:
            base_graph: Base graph to attack
            targeted_node: Optional specific node ID to target

        Returns:
            Adversarial proposal

        Raises:
            ValueError: If base graph invalid
        """
        # Validate input
        valid, error = self.validator.validate_graph(base_graph)
        if not valid:
            raise ValueError(f"Invalid base graph: {error}")

        proposal = copy.deepcopy(base_graph)
        nodes = proposal.get("nodes", [])
        edges = proposal.get("edges", [])
        counter = self._get_counter()

        if not nodes:
            raise ValueError("Cannot create adversarial proposal from empty graph")

        # Adversarial strategies
        adversarial_labels = [
            "ADVERSARY",
            "MALICIOUS",
            "CORRUPTED",
            "POISONED",
            "BACKDOOR",
            "TROJAN",
            "EXPLOIT",
        ]

        target_found = False

        if targeted_node:
            # Target specific node
            for node in nodes:
                if node.get("id") == targeted_node:
                    original_label = node.get("label", "Node")
                    node["label"] = self.rng.choice(adversarial_labels)

                    if "properties" not in node:
                        node["properties"] = {}

                    node["properties"]["adversarial"] = True
                    node["properties"]["original_label"] = original_label
                    node["properties"]["targeted"] = True
                    target_found = True
                    break

            if not target_found:
                logger.warning(
                    f"Targeted node '{targeted_node}' not found, targeting random"
                )

        if not targeted_node or not target_found:
            # Random adversarial mutation
            strategies = ["replace_label", "inject_node", "corrupt_edges"]
            strategy = self.rng.choice(strategies)

            if strategy == "replace_label" and nodes:
                # Replace random node label
                node = self.rng.choice(nodes)
                original_label = node.get("label", "Node")
                node["label"] = self.rng.choice(adversarial_labels)

                if "properties" not in node:
                    node["properties"] = {}

                node["properties"]["adversarial"] = True
                node["properties"]["original_label"] = original_label

            elif strategy == "inject_node":
                # Inject adversarial node
                adv_node_id = f"adv_node_{self.rng.randint(1000, 9999)}_{counter}"
                nodes.append(
                    {
                        "id": adv_node_id,
                        "label": self.rng.choice(adversarial_labels),
                        "properties": {
                            "adversarial": True,
                            "injected": True,
                            "strategy": "injection",
                        },
                    }
                )

                # Connect to random existing nodes
                if len(nodes) > 1:
                    for _ in range(min(3, len(nodes) - 1)):
                        target = self.rng.choice(
                            [n for n in nodes if n["id"] != adv_node_id]
                        )
                        edges.append(
                            {
                                "from": adv_node_id,
                                "to": target["id"],
                                "weight": self.rng.uniform(-1.0, 1.0),
                                "properties": {"adversarial": True, "injected": True},
                            }
                        )

            elif strategy == "corrupt_edges" and edges:
                # Corrupt edge weights
                num_corrupt = min(len(edges), max(1, len(edges) // 3))
                corrupt_edges = self.rng.sample(edges, num_corrupt)

                for edge in corrupt_edges:
                    original_weight = edge.get("weight", 0.5)
                    # Extreme corruption
                    edge["weight"] = self.rng.choice(
                        [float("inf"), float("-inf"), 0.0, 999.0, -999.0]
                    )

                    if "properties" not in edge:
                        edge["properties"] = {}

                    edge["properties"]["adversarial"] = True
                    edge["properties"]["corrupted"] = True
                    edge["properties"]["original_weight"] = original_weight

        # Update metadata
        if "metadata" not in proposal:
            proposal["metadata"] = {}

        proposal["metadata"]["adversarial"] = True
        proposal["metadata"]["targeted_node"] = targeted_node
        proposal["metadata"]["adv_id"] = self._hash_proposal(proposal, "adversarial")
        proposal["metadata"]["generated_at"] = datetime.utcnow().isoformat()

        # Calculate quality (adversarial might have lower quality)
        quality = self._calculate_quality_score(proposal)
        proposal["metadata"]["quality_score"] = quality

        # Check duplicate
        proposal_hash = self._hash_proposal(proposal, "adversarial")
        is_duplicate = self._check_duplicate(proposal_hash)

        if is_duplicate:
            logger.warning(f"Duplicate adversarial proposal: {proposal_hash}")

        # Update metrics and audit
        self._update_metrics("adversarial", quality)
        self._audit("adversarial", proposal, quality)

        with self.lock:
            self.generated_graphs.append(proposal)

        return proposal

    def curriculum_batch(
        self, base_graph: Dict[str, Any], n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generate curriculum learning batch with balanced distribution.

        Args:
            base_graph: Base graph for augmentation
            n: Number of proposals to generate

        Returns:
            List of augmented proposals

        Raises:
            ValueError: If n invalid or base graph invalid
        """
        # Validate input
        valid, error = self.validator.validate_graph(base_graph)
        if not valid:
            raise ValueError(f"Invalid base graph: {error}")

        if not (1 <= n <= MAX_BATCH_SIZE):
            raise ValueError(f"Batch size must be 1-{MAX_BATCH_SIZE}, got {n}")

        batch: List[Dict[str, Any]] = []

        # Calculate balanced distribution
        num_synthetic = n // 3
        num_counterfactual = n // 3
        num_adversarial = n - num_synthetic - num_counterfactual

        logger.info(
            f"Generating curriculum batch: {num_synthetic} synthetic, "
            f"{num_counterfactual} counterfactual, {num_adversarial} adversarial"
        )

        # Generate synthetic
        for i in range(num_synthetic):
            try:
                complexity = self.rng.randint(1, min(3, MAX_COMPLEXITY))
                proposal = self.generate_synthetic_proposal(
                    base_graph, complexity=complexity
                )
                batch.append(proposal)
            except Exception as e:
                logger.error(f"Failed to generate synthetic proposal {i}: {e}")

        # Generate counterfactual
        for i in range(num_counterfactual):
            try:
                invert_all = i % 2 == 0
                proposal = self.counterfactual_proposal(
                    base_graph, invert_all=invert_all
                )
                batch.append(proposal)
            except Exception as e:
                logger.error(f"Failed to generate counterfactual proposal {i}: {e}")

        # Generate adversarial
        for i in range(num_adversarial):
            try:
                proposal = self.adversarial_proposal(base_graph)
                batch.append(proposal)
            except Exception as e:
                logger.error(f"Failed to generate adversarial proposal {i}: {e}")

        logger.info(f"Generated curriculum batch of {len(batch)} proposals")

        return batch

    def get_metrics(self) -> Dict[str, Any]:
        """Get current augmentation metrics."""
        with self.lock:
            return self.metrics.to_dict()

    def get_diversity_score(self) -> float:
        """
        Calculate diversity score of generated proposals.

        Returns:
            Diversity score (0-1)
        """
        with self.lock:
            if len(self.generated_graphs) < 2:
                return 0.0

            # Calculate pairwise diversity
            total_pairs = 0
            diverse_pairs = 0

            for i in range(len(self.generated_graphs)):
                for j in range(i + 1, len(self.generated_graphs)):
                    total_pairs += 1

                    # Simple diversity: different node counts or edge counts
                    g1 = self.generated_graphs[i]
                    g2 = self.generated_graphs[j]

                    n1 = len(g1.get("nodes", []))
                    n2 = len(g2.get("nodes", []))
                    e1 = len(g1.get("edges", []))
                    e2 = len(g2.get("edges", []))

                    if n1 != n2 or e1 != e2:
                        diverse_pairs += 1

            if total_pairs == 0:
                return 0.0

            diversity = diverse_pairs / total_pairs
            self.metrics.diversity_score = diversity

            return diversity

    def reset_metrics(self):
        """Reset all metrics and tracking."""
        with self.lock:
            self.metrics = AugmentationMetrics()
            self.generated_hashes.clear()
            self.generated_graphs.clear()
            self.audit_log.clear()

        logger.info("Metrics reset")


# Demo usage
if __name__ == "__main__":
    print("=" * 60)
    print("Data Augmentor - Production Demo")
    print("=" * 60)

    # Base graph
    base_graph = {
        "graph_id": "demo_graph",
        "nodes": [
            {"id": "n1", "label": "Sentiment", "properties": {}},
            {"id": "n2", "label": "Score", "properties": {}},
            {"id": "n3", "label": "Data", "properties": {}},
        ],
        "edges": [
            {"from": "n1", "to": "n2", "weight": 0.5, "properties": {}},
            {"from": "n2", "to": "n3", "weight": 0.7, "properties": {}},
        ],
        "metadata": {},
    }

    # Create augmentor
    aug = DataAugmentor(random_seed=42)

    # Test 1: Synthetic
    print("\n1. Synthetic Proposal (complexity=2)")
    try:
        synth = aug.generate_synthetic_proposal(base_graph, complexity=2)
        print(f"   Nodes: {len(synth['nodes'])}")
        print(f"   Edges: {len(synth['edges'])}")
        print(f"   Quality: {synth['metadata'].get('quality_score', 0):.3f}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test 2: Counterfactual
    print("\n2. Counterfactual Proposal")
    try:
        cf = aug.counterfactual_proposal(base_graph, invert_all=True)
        print(f"   First node label: {cf['nodes'][0]['label']}")
        print(f"   First edge weight: {cf['edges'][0]['weight']}")
        print(f"   Quality: {cf['metadata'].get('quality_score', 0):.3f}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test 3: Adversarial
    print("\n3. Adversarial Proposal")
    try:
        adv = aug.adversarial_proposal(base_graph, targeted_node="n2")
        node_labels = [n["label"] for n in adv["nodes"]]
        print(f"   Node labels: {node_labels}")
        print(f"   Quality: {adv['metadata'].get('quality_score', 0):.3f}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test 4: Curriculum Batch
    print("\n4. Curriculum Batch (n=9)")
    try:
        batch = aug.curriculum_batch(base_graph, n=9)
        print(f"   Batch size: {len(batch)}")
        types = {}
        for p in batch:
            if p["metadata"].get("synthetic"):
                types["synthetic"] = types.get("synthetic", 0) + 1
            elif p["metadata"].get("counterfactual"):
                types["counterfactual"] = types.get("counterfactual", 0) + 1
            elif p["metadata"].get("adversarial"):
                types["adversarial"] = types.get("adversarial", 0) + 1
        print(f"   Distribution: {types}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test 5: Empty graph handling
    print("\n5. Edge Case - Empty Graph")
    try:
        empty = {"nodes": [], "edges": []}
        aug.generate_synthetic_proposal(empty, complexity=1)
        print("   FAILED - should have raised error")
    except ValueError as e:
        print(f"   PASSED - raised ValueError: {str(e)[:50]}...")

    # Test 6: Metrics
    print("\n6. Metrics Summary")
    metrics = aug.get_metrics()
    print(f"   Total generated: {metrics['total_generated']}")
    print(f"   Synthetic: {metrics['synthetic_count']}")
    print(f"   Counterfactual: {metrics['counterfactual_count']}")
    print(f"   Adversarial: {metrics['adversarial_count']}")
    print(f"   Avg quality: {metrics['avg_quality_score']:.3f}")
    print(f"   Duplicates: {metrics['duplicates_detected']}")

    diversity = aug.get_diversity_score()
    print(f"   Diversity score: {diversity:.3f}")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
