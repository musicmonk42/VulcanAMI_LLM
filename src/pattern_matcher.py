import asyncio
import copy
import json
import logging
import re
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Set

# --- Performance Optimization Import ---
try:
    import networkx as nx
    from networkx.algorithms import isomorphism

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None
    isomorphism = None

# --- Ethical Check Import ---
try:
    from nso_aligner import NSOAligner

    NSO_ALIGNER_AVAILABLE = True
except ImportError:
    NSO_ALIGNER_AVAILABLE = False

    # Create a fail-safe dummy class that REJECTS by default
    class NSOAligner:
        """
        Fail-safe dummy NSOAligner that rejects all proposals by default.
        This ensures safety when the real NSO aligner is not available.
        """

        def multi_model_audit(self, proposal: Any) -> str:
            logging.error(
                "CRITICAL: NSOAligner not found. For safety, ALL rewrites are being REJECTED. "
                "Install nso_aligner module to enable ethical validation."
            )
            # FAIL-SAFE: Reject all proposals when aligner unavailable
            return "risky"


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("PatternMatcher")

# --- Constants ---
MAX_GRAPH_NODES = 10000  # Maximum nodes in a graph
MAX_GRAPH_EDGES = 50000  # Maximum edges in a graph
MAX_PATTERN_NODES = 100  # Maximum nodes in a pattern
DEFAULT_MATCH_TIMEOUT = 30.0  # Seconds
MAX_MATCHES_TO_PROCESS = 1000  # Maximum number of matches to consider
MAX_DEPTH_RECURSION = 50  # Maximum recursion depth for nested structures

# --- Type Definitions for Clarity ---
Graph = Dict[str, List[Dict]]
Pattern = Dict[str, Any]
Match = Dict[str, str]  # Maps pattern node ID (?p1) to graph node ID (n1)
Mutator = Callable[
    [Graph, Match], Graph
]  # A function that rewrites a graph based on a match

# --- Enhanced DSL Logic ---
# Regex to parse simplified DSL strings like "> 10", "<= -5.5", etc.
_DSL_OPERATOR_RE = re.compile(r"^\s*(>|>=|<|<=|==|!=)\s*(-?(?:\d*\.)?\d+)\s*$")


@dataclass
class GraphValidationResult:
    """Result of graph validation."""

    is_valid: bool
    error_message: Optional[str] = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


@dataclass
class MatchingStats:
    """Statistics for pattern matching operations."""

    graphs_processed: int = 0
    patterns_matched: int = 0
    matches_found: int = 0
    rewrites_proposed: int = 0
    rewrites_approved: int = 0
    rewrites_rejected: int = 0
    timeouts: int = 0
    errors: int = 0


class PatternMatcherError(Exception):
    """Base exception for PatternMatcher errors."""

    pass


class GraphValidationError(PatternMatcherError):
    """Raised when graph validation fails."""

    pass


class MatchingTimeoutError(PatternMatcherError):
    """Raised when matching operation times out."""

    pass


class GraphSizeLimitError(PatternMatcherError):
    """Raised when graph exceeds size limits."""

    pass


class PatternMatcher:
    """
    Finds and rewrites subgraph patterns within a larger graph using an enhanced DSL
    and a high-performance NetworkX backend. Integrates ethical checks for safe rewrites.

    Features:
    - Fail-safe ethical validation (rejects when aligner unavailable)
    - Timeout protection for long-running matches
    - Graph size limits and validation
    - Comprehensive error handling
    - Thread-safe statistics tracking
    """

    def __init__(
        self,
        match_timeout: float = DEFAULT_MATCH_TIMEOUT,
        max_matches: int = MAX_MATCHES_TO_PROCESS,
        enable_validation: bool = True,
        strict_mode: bool = True,
    ):
        """
        Initialize PatternMatcher.

        Args:
            match_timeout: Timeout in seconds for matching operations
            max_matches: Maximum number of matches to process
            enable_validation: Enable graph validation
            strict_mode: Enable strict validation and type checking
        """
        if not NETWORKX_AVAILABLE:
            raise ImportError(
                "NetworkX is required for the optimized PatternMatcher. "
                "Please install it using 'pip install networkx'."
            )

        self.match_timeout = match_timeout
        self.max_matches = max_matches
        self.enable_validation = enable_validation
        self.strict_mode = strict_mode

        # Initialize NSO Aligner
        self.nso_aligner = NSOAligner()

        # Statistics tracking
        self.stats = MatchingStats()
        self.stats_lock = threading.Lock()

        # Warn if NSO Aligner is not available
        if not NSO_ALIGNER_AVAILABLE:
            logger.critical(
                "⚠️  NSO_ALIGNER NOT AVAILABLE - Operating in FAIL-SAFE mode. "
                "All rewrites will be REJECTED until nso_aligner module is installed."
            )

        logger.info(
            f"PatternMatcher initialized with NetworkX backend "
            f"(timeout={match_timeout}s, max_matches={max_matches}, "
            f"validation={'enabled' if enable_validation else 'disabled'}, "
            f"strict={'on' if strict_mode else 'off'})"
        )

    def _validate_graph_structure(
        self, data: Dict, graph_type: str = "graph"
    ) -> GraphValidationResult:
        """
        Validate graph structure for correctness and size limits.

        Args:
            data: Graph dictionary to validate
            graph_type: Type of graph ("graph" or "pattern")

        Returns:
            GraphValidationResult with validation status
        """
        warnings = []

        # Check basic structure
        if not isinstance(data, dict):
            return GraphValidationResult(
                False, f"{graph_type} must be a dictionary, got {type(data).__name__}"
            )

        if "nodes" not in data:
            return GraphValidationResult(False, f"{graph_type} missing 'nodes' field")

        if not isinstance(data["nodes"], list):
            return GraphValidationResult(
                False,
                f"{graph_type} 'nodes' must be a list, got {type(data['nodes']).__name__}",
            )

        nodes = data["nodes"]
        edges = data.get("edges", [])

        # Check size limits
        max_nodes = MAX_PATTERN_NODES if graph_type == "pattern" else MAX_GRAPH_NODES
        if len(nodes) > max_nodes:
            return GraphValidationResult(
                False,
                f"{graph_type} has {len(nodes)} nodes, exceeds limit of {max_nodes}",
            )

        if len(edges) > MAX_GRAPH_EDGES:
            return GraphValidationResult(
                False,
                f"{graph_type} has {len(edges)} edges, exceeds limit of {MAX_GRAPH_EDGES}",
            )

        # Validate nodes
        node_ids = set()
        for i, node in enumerate(nodes):
            if not isinstance(node, dict):
                return GraphValidationResult(
                    False, f"{graph_type} node at index {i} is not a dictionary"
                )

            if "id" not in node:
                return GraphValidationResult(
                    False, f"{graph_type} node at index {i} missing 'id' field"
                )

            node_id = node["id"]
            if not isinstance(node_id, str):
                return GraphValidationResult(
                    False,
                    f"{graph_type} node at index {i} has non-string id: {type(node_id).__name__}",
                )

            if node_id in node_ids:
                return GraphValidationResult(
                    False, f"{graph_type} has duplicate node id: {node_id}"
                )

            node_ids.add(node_id)

            if "type" not in node:
                return GraphValidationResult(
                    False, f"{graph_type} node '{node_id}' missing 'type' field"
                )

            # Validate params if present
            if "params" in node and not isinstance(node["params"], dict):
                return GraphValidationResult(
                    False, f"{graph_type} node '{node_id}' has non-dict params"
                )

        # Validate edges
        if not isinstance(edges, list):
            return GraphValidationResult(
                False,
                f"{graph_type} 'edges' must be a list, got {type(edges).__name__}",
            )

        for i, edge in enumerate(edges):
            if not isinstance(edge, dict):
                return GraphValidationResult(
                    False, f"{graph_type} edge at index {i} is not a dictionary"
                )

            if "from" not in edge:
                return GraphValidationResult(
                    False, f"{graph_type} edge at index {i} missing 'from' field"
                )

            if "to" not in edge:
                return GraphValidationResult(
                    False, f"{graph_type} edge at index {i} missing 'to' field"
                )

            from_id = edge["from"]
            to_id = edge["to"]

            if from_id not in node_ids:
                return GraphValidationResult(
                    False,
                    f"{graph_type} edge {i} references non-existent 'from' node: {from_id}",
                )

            if to_id not in node_ids:
                return GraphValidationResult(
                    False,
                    f"{graph_type} edge {i} references non-existent 'to' node: {to_id}",
                )

        # Warnings for potentially problematic structures
        if len(nodes) == 0:
            warnings.append(f"{graph_type} has no nodes")

        if len(edges) == 0 and len(nodes) > 1:
            warnings.append(f"{graph_type} has multiple nodes but no edges")

        # Check for isolated nodes
        nodes_with_edges = set()
        for edge in edges:
            nodes_with_edges.add(edge["from"])
            nodes_with_edges.add(edge["to"])

        isolated = node_ids - nodes_with_edges
        if isolated and len(nodes) > 1:
            warnings.append(f"{graph_type} has {len(isolated)} isolated node(s)")

        return GraphValidationResult(True, None, warnings)

    def _to_networkx_graph(self, data: Graph) -> nx.DiGraph:
        """
        Converts the dictionary-based graph representation to a NetworkX DiGraph.

        Args:
            data: Graph dictionary

        Returns:
            NetworkX DiGraph
        """
        G = nx.DiGraph()

        # Add nodes with all attributes
        for node in data["nodes"]:
            node_id = node["id"]
            # Store all node data as attributes
            G.add_node(node_id, **node)

        # Add edges
        for edge in data.get("edges", []):
            G.add_edge(edge["from"], edge["to"])

        return G

    def _safe_type_cast(self, value: Any, target_type: type, param_name: str) -> Any:
        """
        Safely cast a value to a target type with comprehensive error handling.

        Args:
            value: Value to cast
            target_type: Target type
            param_name: Parameter name for error messages

        Returns:
            Casted value

        Raises:
            ValueError: If casting fails
        """
        # Handle same-type case
        if type(value) == target_type:
            return value

        # Handle numeric types
        if target_type in (int, float):
            try:
                return target_type(value)
            except (ValueError, TypeError, OverflowError) as e:
                raise ValueError(
                    f"Cannot cast '{value}' ({type(value).__name__}) to "
                    f"{target_type.__name__} for parameter '{param_name}': {e}"
                )

        # Handle string type
        if target_type == str:
            return str(value)

        # Handle bool type
        if target_type == bool:
            if isinstance(value, str):
                value_lower = value.lower()
                if value_lower in ("true", "1", "yes", "on"):
                    return True
                if value_lower in ("false", "0", "no", "off"):
                    return False
            return bool(value)

        # Unsupported type
        raise ValueError(
            f"Unsupported type cast from {type(value).__name__} to "
            f"{target_type.__name__} for parameter '{param_name}'"
        )

    def _node_semantic_match(self, g_node_attrs: Dict, p_node_attrs: Dict) -> bool:
        """
        The core semantic comparison function passed to NetworkX.
        Checks if a graph node matches a pattern node based on the enhanced DSL.

        Args:
            g_node_attrs: Graph node attributes
            p_node_attrs: Pattern node attributes

        Returns:
            True if nodes match, False otherwise
        """
        try:
            # 1. Check node type (with wildcard support)
            p_type = p_node_attrs.get("type")
            g_type = g_node_attrs.get("type")

            if p_type != "*" and p_type != g_type:
                return False

            # 2. Check parameter constraints
            p_params = p_node_attrs.get("params", {})
            g_params = g_node_attrs.get("params", {})

            for p_key, p_constraint in p_params.items():
                # Graph node must have the parameter
                if p_key not in g_params:
                    return False

                g_value = g_params[p_key]

                # A. Enhanced DSL: Handle string-based operators like "> 10"
                if isinstance(p_constraint, str):
                    match = _DSL_OPERATOR_RE.match(p_constraint)
                    if match:
                        op, p_val_str = match.groups()

                        # Safe type casting
                        try:
                            p_val = self._safe_type_cast(
                                p_val_str, type(g_value), p_key
                            )
                        except ValueError as e:
                            if self.strict_mode:
                                logger.warning(
                                    f"Type cast failed for parameter '{p_key}': {e}"
                                )
                                return False
                            else:
                                # In non-strict mode, try string comparison
                                p_val = p_val_str

                        # Perform comparison
                        try:
                            if op == ">" and not g_value > p_val:
                                return False
                            if op == ">=" and not g_value >= p_val:
                                return False
                            if op == "<" and not g_value < p_val:
                                return False
                            if op == "<=" and not g_value <= p_val:
                                return False
                            if op == "==" and not g_value == p_val:
                                return False
                            if op == "!=" and not g_value != p_val:
                                return False
                        except TypeError as e:
                            logger.warning(
                                f"Comparison failed for parameter '{p_key}': {e}"
                            )
                            return False

                        continue  # Move to the next parameter constraint

                    # Fallback to simple string equality if it's not a DSL operator string
                    elif g_value != p_constraint:
                        return False

                # B. Original DSL: Handle dictionary-based operators like {">": 10}
                elif isinstance(p_constraint, dict):
                    if not p_constraint:  # Empty constraint dict
                        logger.warning(f"Empty constraint dict for parameter '{p_key}'")
                        return False

                    for op, p_val in p_constraint.items():
                        try:
                            if op == ">":
                                if not g_value > p_val:
                                    return False
                            elif op == ">=":
                                if not g_value >= p_val:
                                    return False
                            elif op == "<":
                                if not g_value < p_val:
                                    return False
                            elif op == "<=":
                                if not g_value <= p_val:
                                    return False
                            elif op == "==":
                                if not g_value == p_val:
                                    return False
                            elif op == "!=":
                                if not g_value != p_val:
                                    return False
                            elif op == "exists":
                                expected_exists = bool(p_val)
                                actual_exists = g_value is not None
                                if expected_exists != actual_exists:
                                    return False
                            else:
                                logger.warning(
                                    f"Unknown operator '{op}' for parameter '{p_key}'"
                                )
                                if self.strict_mode:
                                    return False
                        except (TypeError, ValueError) as e:
                            logger.warning(
                                f"Operator '{op}' failed for parameter '{p_key}': {e}"
                            )
                            return False

                # C. Simple direct value comparison
                elif g_value != p_constraint:
                    return False

            return True  # All checks passed

        except Exception as e:
            logger.error(f"Unexpected error in semantic matching: {e}", exc_info=True)
            return False

    async def find_matches(
        self, graph: Graph, pattern: Pattern
    ) -> AsyncGenerator[Match, None]:
        """
        Asynchronously finds all occurrences of a pattern in a graph using a parallel-capable backend.

        Args:
            graph: Graph to search in
            pattern: Pattern to search for

        Yields:
            A dictionary representing a single match, mapping pattern node IDs to graph node IDs.

        Raises:
            GraphValidationError: If validation fails
            MatchingTimeoutError: If matching times out
        """
        # Validate inputs if enabled
        if self.enable_validation:
            graph_validation = self._validate_graph_structure(graph, "graph")
            if not graph_validation.is_valid:
                raise GraphValidationError(
                    f"Graph validation failed: {graph_validation.error_message}"
                )

            for warning in graph_validation.warnings:
                logger.warning(f"Graph validation warning: {warning}")

            pattern_validation = self._validate_graph_structure(pattern, "pattern")
            if not pattern_validation.is_valid:
                raise GraphValidationError(
                    f"Pattern validation failed: {pattern_validation.error_message}"
                )

            for warning in pattern_validation.warnings:
                logger.warning(f"Pattern validation warning: {warning}")

        try:
            # Convert to NetworkX graphs
            G = self._to_networkx_graph(graph)
            P = self._to_networkx_graph(pattern)

            # Use NetworkX's optimized DiGraphMatcher for subgraph isomorphism
            matcher = isomorphism.DiGraphMatcher(
                G, P, node_match=self._node_semantic_match
            )

            # Wrap the matching process with timeout protection
            loop = asyncio.get_running_loop()

            async def find_with_timeout():
                """Find matches with timeout protection."""
                try:
                    # Run in executor to avoid blocking event loop
                    match_iter = await asyncio.wait_for(
                        loop.run_in_executor(
                            None, list, matcher.subgraph_isomorphisms_iter()
                        ),
                        timeout=self.match_timeout,
                    )
                    return match_iter
                except asyncio.TimeoutError:
                    raise MatchingTimeoutError(
                        f"Pattern matching timed out after {self.match_timeout} seconds. "
                        f"Graph may be too large or pattern too complex."
                    )

            # Get matches with timeout
            matches = await find_with_timeout()

            # Update statistics
            with self.stats_lock:
                self.stats.patterns_matched += 1
                self.stats.matches_found += len(matches)

            # Yield matches up to max_matches limit
            # NetworkX returns matches as {graph_node_id: pattern_node_id}
            # We need to invert to {pattern_node_id: graph_node_id}
            for i, match in enumerate(matches):
                if i >= self.max_matches:
                    logger.warning(
                        f"Reached maximum match limit ({self.max_matches}). "
                        f"Stopping after processing {i} matches."
                    )
                    break

                # Invert the match dictionary
                inverted_match = {
                    pattern_id: graph_id for graph_id, pattern_id in match.items()
                }
                yield inverted_match

        except MatchingTimeoutError:
            with self.stats_lock:
                self.stats.timeouts += 1
            raise
        except Exception as e:
            with self.stats_lock:
                self.stats.errors += 1
            logger.error(f"Error during pattern matching: {e}", exc_info=True)
            raise

    async def rewrite_graph(
        self, graph: Graph, pattern: Pattern, mutator: Mutator
    ) -> Graph:
        """
        Finds matches for a pattern, proposes rewrites via a mutator, validates them
        for ethical alignment, and applies the first approved rewrite.

        Args:
            graph: Graph to rewrite
            pattern: Pattern to match
            mutator: Function that generates rewrite proposals

        Returns:
            Rewritten graph, or original graph if no approved rewrites found

        Raises:
            GraphValidationError: If validation fails
            MatchingTimeoutError: If matching times out
        """
        # Update statistics
        with self.stats_lock:
            self.stats.graphs_processed += 1

        try:
            async for match in self.find_matches(graph, pattern):
                logger.info(f"Found potential match: {match}. Proposing rewrite.")

                # Generate proposed rewrite
                try:
                    proposed_graph = mutator(graph, match)
                except Exception as e:
                    logger.error(f"Mutator failed: {e}", exc_info=True)
                    with self.stats_lock:
                        self.stats.errors += 1
                    continue

                # Validate proposed graph structure
                if self.enable_validation:
                    validation = self._validate_graph_structure(
                        proposed_graph, "proposed_graph"
                    )
                    if not validation.is_valid:
                        logger.error(
                            f"Proposed graph is invalid: {validation.error_message}. "
                            "Rejecting rewrite."
                        )
                        with self.stats_lock:
                            self.stats.rewrites_rejected += 1
                        continue

                # Update statistics
                with self.stats_lock:
                    self.stats.rewrites_proposed += 1

                # Ethical validation
                logger.info("Validating proposed rewrite with NSO Aligner...")
                try:
                    audit_label = self.nso_aligner.multi_model_audit(proposed_graph)
                except Exception as e:
                    logger.error(f"NSO Aligner failed: {e}", exc_info=True)
                    # Fail-safe: reject on error
                    audit_label = "risky"

                if audit_label not in ("safe", "unknown"):
                    logger.warning(
                        f"Rewrite REJECTED by NSO Aligner. Reason: '{audit_label}'. "
                        "Discarding change and searching for other potential matches."
                    )
                    with self.stats_lock:
                        self.stats.rewrites_rejected += 1
                    continue

                logger.info("Rewrite APPROVED by NSO Aligner. Applying changes.")
                with self.stats_lock:
                    self.stats.rewrites_approved += 1

                return proposed_graph  # Return the first approved graph

            logger.warning(
                "No approved matches found for the given pattern. Graph remains unchanged."
            )
            return graph

        except (GraphValidationError, MatchingTimeoutError):
            # Re-raise these specific exceptions
            raise
        except Exception as e:
            logger.error(f"Error during graph rewriting: {e}", exc_info=True)
            with self.stats_lock:
                self.stats.errors += 1
            raise

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current statistics.

        Returns:
            Dictionary with statistics
        """
        with self.stats_lock:
            return {
                "graphs_processed": self.stats.graphs_processed,
                "patterns_matched": self.stats.patterns_matched,
                "matches_found": self.stats.matches_found,
                "rewrites_proposed": self.stats.rewrites_proposed,
                "rewrites_approved": self.stats.rewrites_approved,
                "rewrites_rejected": self.stats.rewrites_rejected,
                "approval_rate": (
                    self.stats.rewrites_approved / self.stats.rewrites_proposed
                    if self.stats.rewrites_proposed > 0
                    else 0.0
                ),
                "timeouts": self.stats.timeouts,
                "errors": self.stats.errors,
                "nso_aligner_available": NSO_ALIGNER_AVAILABLE,
            }

    def reset_statistics(self):
        """Reset statistics counters."""
        with self.stats_lock:
            self.stats = MatchingStats()
        logger.info("Statistics reset")


# --- Example Usage ---
async def main():
    """Demonstrates the PatternMatcher's capabilities."""

    print("\n" + "=" * 70)
    print("PatternMatcher Production Demo")
    print("=" * 70 + "\n")

    # A sample graph for our AI agents to operate on
    target_graph = {
        "nodes": [
            {"id": "n1", "type": "CONST", "params": {"value": 5}},
            {"id": "n2", "type": "CONST", "params": {"value": 20}},
            {"id": "n3", "type": "ADD", "params": {}},
            {"id": "n4", "type": "CONST", "params": {"value": 50}},
        ],
        "edges": [{"from": "n1", "to": "n3"}, {"from": "n2", "to": "n3"}],
    }

    # Initialize matcher with production settings
    matcher = PatternMatcher(
        match_timeout=10.0, max_matches=100, enable_validation=True, strict_mode=True
    )

    # --- 1. Demonstrate finding matches with the ENHANCED DSL ---
    logger.info(
        "\n--- Test 1: Finding all CONST nodes with value > 10 (using new DSL shorthand) ---"
    )

    # This pattern uses the simplified DSL string "> 10"
    pattern_to_find = {
        "nodes": [{"id": "?p1", "type": "CONST", "params": {"value": "> 10"}}],
        "edges": [],
    }

    found_count = 0
    try:
        async for match in matcher.find_matches(target_graph, pattern_to_find):
            found_count += 1
            logger.info(
                f"Found match: Pattern node '?p1' maps to graph node '{match['?p1']}'"
            )
    except Exception as e:
        logger.error(f"Error during matching: {e}")

    if found_count == 0:
        logger.info("No matches found.")
    else:
        logger.info(f"Total matches found: {found_count}")

    # --- 2. Demonstrate rewriting with an APPROVED mutator ---
    logger.info("\n--- Test 2: Rewriting a matched node (approved by Aligner) ---")

    def safe_optimizer_mutator(graph: Graph, match: Match) -> Graph:
        """This mutator proposes a safe change (doubling the value)."""
        new_graph = copy.deepcopy(graph)
        node_to_change_id = match["?p1"]

        for node in new_graph["nodes"]:
            if node["id"] == node_to_change_id:
                original_value = node["params"]["value"]
                node["params"]["value"] = original_value * 2
                logger.info(
                    f"Mutator proposing change on node '{node_to_change_id}': "
                    f"value {original_value} -> {node['params']['value']}"
                )
                break
        return new_graph

    try:
        approved_graph = await matcher.rewrite_graph(
            target_graph, pattern_to_find, safe_optimizer_mutator
        )

        logger.info("\nOriginal graph for Test 2:")
        print(json.dumps(target_graph["nodes"], indent=2))
        logger.info("\nGraph after approved rewrite:")
        print(json.dumps(approved_graph["nodes"], indent=2))
    except Exception as e:
        logger.error(f"Error during rewrite: {e}")

    # --- 3. Demonstrate a REJECTED rewrite due to ethical checks (only if aligner available) ---
    if NSO_ALIGNER_AVAILABLE:
        logger.info(
            "\n--- Test 3: Attempting a rewrite (may be rejected by Aligner) ---"
        )

        def risky_optimizer_mutator(graph: Graph, match: Match) -> Graph:
            """This mutator proposes a potentially risky change."""
            new_graph = copy.deepcopy(graph)
            node_to_change_id = match["?p1"]

            for node in new_graph["nodes"]:
                if node["id"] == node_to_change_id:
                    original_value = node["params"]["value"]
                    # A nonsensical or potentially harmful value
                    node["params"]["value"] = -999
                    logger.info(
                        f"Mutator proposing RISKY change on node '{node_to_change_id}': "
                        f"value {original_value} -> {node['params']['value']}"
                    )
                    break
            return new_graph

        try:
            rejected_graph = await matcher.rewrite_graph(
                target_graph, pattern_to_find, risky_optimizer_mutator
            )

            logger.info("\nOriginal graph for Test 3:")
            print(json.dumps(target_graph["nodes"], indent=2))
            logger.info("\nGraph after potentially rejected rewrite:")
            print(json.dumps(rejected_graph["nodes"], indent=2))
        except Exception as e:
            logger.error(f"Error during rewrite: {e}")
    else:
        logger.warning(
            "\n--- Test 3: SKIPPED (NSO Aligner not available - all rewrites are auto-rejected) ---"
        )

    # --- 4. Demonstrate validation errors ---
    logger.info("\n--- Test 4: Testing validation (invalid graph) ---")

    invalid_graph = {
        "nodes": [
            {"id": "n1", "type": "CONST", "params": {"value": 10}},
            {"id": "n1", "type": "CONST", "params": {"value": 20}},  # Duplicate ID
        ],
        "edges": [],
    }

    invalid_pattern = {
        "nodes": [{"id": "?p1", "type": "CONST", "params": {"value": "> 5"}}],
        "edges": [],
    }

    try:
        async for match in matcher.find_matches(invalid_graph, invalid_pattern):
            logger.info(f"Found match: {match}")
    except GraphValidationError as e:
        logger.info(f"✓ Validation correctly caught error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

    # --- 5. Display statistics ---
    logger.info("\n--- Test 5: Statistics Summary ---")
    stats = matcher.get_statistics()
    print("\nPattern Matching Statistics:")
    print(f"  Graphs processed: {stats['graphs_processed']}")
    print(f"  Patterns matched: {stats['patterns_matched']}")
    print(f"  Matches found: {stats['matches_found']}")
    print(f"  Rewrites proposed: {stats['rewrites_proposed']}")
    print(f"  Rewrites approved: {stats['rewrites_approved']}")
    print(f"  Rewrites rejected: {stats['rewrites_rejected']}")
    print(f"  Approval rate: {stats['approval_rate']:.1%}")
    print(f"  Timeouts: {stats['timeouts']}")
    print(f"  Errors: {stats['errors']}")
    print(f"  NSO Aligner available: {stats['nso_aligner_available']}")

    print("\n" + "=" * 70)
    print("Demo Complete")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # Check for NetworkX before running
    if not NETWORKX_AVAILABLE:
        logger.error(
            "This script requires the 'networkx' library. "
            "Please install it with 'pip install networkx' and run again."
        )
        sys.exit(1)
    else:
        asyncio.run(main())
