"""
IR Generator for Testing
========================
Generates comprehensive set of malicious/invalid Graphix IR graphs for stress testing.
Security-hardened version with obfuscated patterns and proper error handling.
"""

import json
import random
import string
import os
import base64
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict

# Configuration constants
DEFAULT_OVERSIZED_NODES = 1000
DEFAULT_OVERSIZED_EDGES = 2000
DEFAULT_BUFFER_SIZE_KB = 100
DEFAULT_NESTING_DEPTH = 50
MAX_SAFE_RECURSION_DEPTH = 50

class PatternGenerator:
    """Generates obfuscated test patterns to avoid creating weaponizable exploits."""
    
    @staticmethod
    def get_sql_pattern(index: int = 0) -> str:
        """Generate SQL injection test pattern."""
        patterns = [
            "' OR '1'='1",
            "UNION SELECT",
            "DROP_TABLE_PATTERN",
            "'; EXEC_CMD --"
        ]
        return patterns[index % len(patterns)]
    
    @staticmethod
    def get_command_pattern(index: int = 0) -> str:
        """Generate command injection test pattern."""
        patterns = [
            "; COMMAND_HERE",
            "&& COMMAND_HERE",
            "| PIPE_COMMAND",
            "`BACKTICK_CMD`"
        ]
        return patterns[index % len(patterns)]
    
    @staticmethod
    def get_xss_pattern(index: int = 0) -> str:
        """Generate XSS test pattern."""
        patterns = [
            "<SCRIPT_TAG>alert()</SCRIPT_TAG>",
            "javascript:void(0)",
            "onerror=HANDLER",
            "onload=HANDLER"
        ]
        return patterns[index % len(patterns)]
    
    @staticmethod
    def get_path_pattern(index: int = 0) -> str:
        """Generate path traversal test pattern."""
        patterns = [
            "../PARENT_DIR",
            "..\\PARENT_DIR",
            "/etc/CONFIG_FILE",
            "C:\\Windows\\SYSTEM_FILE"
        ]
        return patterns[index % len(patterns)]
    
    @staticmethod
    def get_template_pattern(index: int = 0) -> str:
        """Generate template injection test pattern."""
        patterns = [
            "{{TEMPLATE_VAR}}",
            "{%TEMPLATE_CMD%}",
            "${TEMPLATE_EXPR}",
            "#{TEMPLATE_REF}"
        ]
        return patterns[index % len(patterns)]

class IRGenerator:
    """
    Comprehensive generator for invalid/malicious IR graphs for stress testing.
    Security-hardened with obfuscated patterns and proper error handling.
    """
    
    def __init__(self, seed: Optional[int] = None, verbose: bool = False):
        """
        Initialize the IR generator with optional seed for reproducibility.
        
        Args:
            seed: Random seed for reproducible test generation
            verbose: Whether to print detailed generation information
        """
        self.verbose = verbose
        if seed is not None:
            random.seed(seed)
            self.seed = seed
        else:
            self.seed = random.randint(0, 1000000)
            random.seed(self.seed)
        
        self.generated_count = 0
        self.output_dir = "test_ir"
        
        # Track generated IDs for reference testing
        self.valid_node_ids = []
        self.invalid_node_ids = []
        
        # Pattern generator for security-safe test patterns
        self.patterns = PatternGenerator()
        
        # Configuration
        self.oversized_nodes = DEFAULT_OVERSIZED_NODES
        self.oversized_edges = DEFAULT_OVERSIZED_EDGES
        self.buffer_size_kb = DEFAULT_BUFFER_SIZE_KB
        self.nesting_depth = DEFAULT_NESTING_DEPTH
    
    def configure(self, oversized_nodes: int = None, oversized_edges: int = None,
                 buffer_size_kb: int = None, nesting_depth: int = None):
        """Configure generator parameters."""
        if oversized_nodes is not None:
            self.oversized_nodes = oversized_nodes
        if oversized_edges is not None:
            self.oversized_edges = oversized_edges
        if buffer_size_kb is not None:
            self.buffer_size_kb = buffer_size_kb
        if nesting_depth is not None:
            self.nesting_depth = min(nesting_depth, MAX_SAFE_RECURSION_DEPTH)
    
    def generate_invalid_graph(self) -> Dict[str, Any]:
        """
        Generate a random invalid graph with various types of malicious patterns.
        Randomly selects from multiple invalid graph generation strategies.
        """
        strategies = [
            self._generate_missing_fields_graph,
            self._generate_invalid_types_graph,
            self._generate_circular_dependency_graph,
            self._generate_duplicate_ids_graph,
            self._generate_invalid_references_graph,
            self._generate_buffer_overflow_graph,
            self._generate_injection_attempt_graph,
            self._generate_deeply_nested_graph,
            self._generate_null_fields_graph,
            self._generate_empty_graph,
            self._generate_unicode_exploit_graph,
            self._generate_type_confusion_graph,
            self._generate_oversized_graph,
            self._generate_malformed_json_graph
        ]
        
        strategy = random.choice(strategies)
        graph = strategy()
        self.generated_count += 1
        
        if self.verbose:
            print(f"Generated {strategy.__name__} graph (#{self.generated_count})")
        
        return graph
    
    def _generate_missing_fields_graph(self) -> Dict[str, Any]:
        """Generate a graph with missing required fields - deterministic removal."""
        graph_id = self._generate_id("missing_fields")
        
        # Create base graph with intentional missing fields
        base_graph = {
            "grammar_version": "1.0.0",
            "id": graph_id,
            "type": "Graph",
            "nodes": [
                {"id": "node1", "type": "InvalidType"},
                {"id": "node2"}  # Missing type
            ],
            "edges": [
                {"from": "node1", "to": "node3"},  # Missing node3
                {"to": "node2"}  # Missing from field
            ]
        }
        
        # Deterministically remove fields based on seed
        fields_to_remove = ["grammar_version", "type", "nodes", "edges"]
        # Use generator count for determinism
        remove_count = (self.generated_count % 2) + 1
        remove_indices = [(self.generated_count + i) % len(fields_to_remove) for i in range(remove_count)]
        
        for idx in sorted(remove_indices, reverse=True):
            field = fields_to_remove[idx]
            if field in base_graph:
                del base_graph[field]
        
        return base_graph
    
    def _generate_invalid_types_graph(self) -> Dict[str, Any]:
        """Generate a graph with invalid data types for fields."""
        graph_id = self._generate_id("invalid_types")
        
        # Rotate through different invalid type combinations
        type_index = self.generated_count % 4
        
        if type_index == 0:
            return {
                "grammar_version": 1.0,  # Number instead of string
                "id": graph_id,
                "type": 123,  # Number instead of string
                "nodes": "not_an_array",
                "edges": {"edge1": {"from": "a", "to": "b"}}
            }
        elif type_index == 1:
            return {
                "grammar_version": ["1", "0", "0"],  # Array instead of string
                "id": 12345,  # Number instead of string
                "type": True,  # Boolean instead of string
                "nodes": [{"id": 123, "type": True}],
                "edges": [{"from": 123, "to": False}]
            }
        elif type_index == 2:
            return {
                "grammar_version": None,
                "id": graph_id,
                "type": "Graph",
                "nodes": [{"id": ["n1"], "type": {"nested": "obj"}}],
                "edges": [{"from": True, "to": None}]
            }
        else:
            return {
                "grammar_version": "1.0.0",
                "id": {"complex": "id"},
                "type": [["nested", "array"]],
                "nodes": 999,
                "edges": "string_edges"
            }
    
    def _generate_circular_dependency_graph(self) -> Dict[str, Any]:
        """Generate a graph with circular dependencies including complex cycles."""
        graph_id = self._generate_id("circular")
        
        nodes = []
        edges = []
        
        # Simple self-loop
        nodes.append({"id": "self_loop", "type": "ComputeNode"})
        edges.append({"from": "self_loop", "to": "self_loop"})
        
        # Two-node cycle
        nodes.extend([
            {"id": "cycle_a", "type": "ComputeNode"},
            {"id": "cycle_b", "type": "ComputeNode"}
        ])
        edges.extend([
            {"from": "cycle_a", "to": "cycle_b"},
            {"from": "cycle_b", "to": "cycle_a"}
        ])
        
        # Three-node cycle
        nodes.extend([
            {"id": "tri_1", "type": "ComputeNode"},
            {"id": "tri_2", "type": "ComputeNode"},
            {"id": "tri_3", "type": "ComputeNode"}
        ])
        edges.extend([
            {"from": "tri_1", "to": "tri_2"},
            {"from": "tri_2", "to": "tri_3"},
            {"from": "tri_3", "to": "tri_1"}
        ])
        
        # Complex cycle with multiple paths (5-node strongly connected component)
        nodes.extend([
            {"id": "scc_1", "type": "ComputeNode"},
            {"id": "scc_2", "type": "ComputeNode"},
            {"id": "scc_3", "type": "ComputeNode"},
            {"id": "scc_4", "type": "ComputeNode"},
            {"id": "scc_5", "type": "ComputeNode"}
        ])
        edges.extend([
            {"from": "scc_1", "to": "scc_2"},
            {"from": "scc_2", "to": "scc_3"},
            {"from": "scc_3", "to": "scc_4"},
            {"from": "scc_4", "to": "scc_5"},
            {"from": "scc_5", "to": "scc_1"},
            {"from": "scc_1", "to": "scc_3"},  # Shortcut
            {"from": "scc_3", "to": "scc_5"},  # Another shortcut
        ])
        
        # Nested cycles (two cycles sharing a node)
        nodes.extend([
            {"id": "shared", "type": "ComputeNode"},
            {"id": "cycle1_node", "type": "ComputeNode"},
            {"id": "cycle2_node", "type": "ComputeNode"}
        ])
        edges.extend([
            {"from": "shared", "to": "cycle1_node"},
            {"from": "cycle1_node", "to": "shared"},
            {"from": "shared", "to": "cycle2_node"},
            {"from": "cycle2_node", "to": "shared"}
        ])
        
        return {
            "grammar_version": "1.0.0",
            "id": graph_id,
            "type": "Graph",
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "contains_cycles": True,
                "cycle_types": ["self_loop", "2_node", "3_node", "complex_scc", "nested"],
                "cycle_count": 6
            }
        }
    
    def _generate_duplicate_ids_graph(self) -> Dict[str, Any]:
        """Generate a graph with duplicate node and edge IDs."""
        graph_id = self._generate_id("duplicates")
        
        duplicate_id = "duplicate_node"
        
        # Multiple unique nodes plus duplicates
        nodes = [
            {"id": duplicate_id, "type": "InputNode"},
            {"id": duplicate_id, "type": "OutputNode"},  # Same ID
            {"id": duplicate_id, "type": "ComputeNode"},  # Same ID again
            {"id": "unique_node_1", "type": "ComputeNode"},
            {"id": "unique_node_2", "type": "TransformNode"},
            {"id": "unique_node_3", "type": "AggregateNode"},
            {"id": "unique_node_1", "type": "FilterNode"},  # Duplicate of unique_node_1
        ]
        
        edges = [
            {"id": "edge1", "from": duplicate_id, "to": "unique_node_1"},
            {"id": "edge1", "from": "unique_node_1", "to": duplicate_id},  # Duplicate edge ID
            {"id": "edge2", "from": "unique_node_2", "to": "unique_node_3"},
            {"id": "edge1", "from": "unique_node_3", "to": "unique_node_2"},  # Another duplicate
        ]
        
        return {
            "grammar_version": "1.0.0",
            "id": graph_id,
            "type": "Graph",
            "nodes": nodes,
            "edges": edges
        }
    
    def _generate_invalid_references_graph(self) -> Dict[str, Any]:
        """Generate a graph with edges referencing non-existent nodes."""
        graph_id = self._generate_id("invalid_refs")
        
        nodes = [
            {"id": "existing_1", "type": "InputNode"},
            {"id": "existing_2", "type": "OutputNode"},
            {"id": "existing_3", "type": "ComputeNode"}
        ]
        
        # Test various types of invalid references
        edges = [
            {"from": "existing_1", "to": "non_existent_1"},
            {"from": "non_existent_2", "to": "existing_2"},
            {"from": "non_existent_3", "to": "non_existent_4"},
            {"from": "existing_1", "to": "existing_2"},  # One valid edge
            {"from": "", "to": "existing_1"},  # Empty string reference
            {"from": "existing_3", "to": ""},  # Empty target
            {"from": "   ", "to": "existing_2"},  # Whitespace-only reference
            {"from": "existing_1", "to": "EXISTING_1"},  # Case-sensitive mismatch
        ]
        
        return {
            "grammar_version": "1.0.0",
            "id": graph_id,
            "type": "Graph",
            "nodes": nodes,
            "edges": edges
        }
    
    def _generate_buffer_overflow_graph(self) -> Dict[str, Any]:
        """Generate a graph designed to test buffer overflow vulnerabilities."""
        graph_id = self._generate_id("buffer_overflow")
        
        # Create configurable large strings
        buffer_size = self.buffer_size_kb * 1024
        overflow_string = "A" * buffer_size
        long_id = "node_" + "x" * (buffer_size // 10)
        
        nodes = [
            {"id": long_id, "type": "ComputeNode"},
            {"id": "normal", "type": overflow_string},
            {
                "id": "nested_overflow",
                "type": "ComputeNode",
                "data": overflow_string,
                "metadata": {
                    "description": overflow_string,
                    "nested": {
                        "deeply": {
                            "nested": {
                                "value": overflow_string
                            }
                        }
                    }
                }
            },
            {
                "id": "array_overflow",
                "type": "Node",
                "large_array": ["x" * 1000 for _ in range(100)]
            }
        ]
        
        edges = [
            {"from": long_id, "to": "normal", "label": overflow_string}
        ]
        
        return {
            "grammar_version": "1.0.0",
            "id": graph_id,
            "type": "Graph",
            "nodes": nodes,
            "edges": edges,
            "description": overflow_string,
            "metadata": {
                "buffer_size_kb": self.buffer_size_kb,
                "test_type": "buffer_overflow"
            }
        }
    
    def _generate_injection_attempt_graph(self) -> Dict[str, Any]:
        """Generate a graph with various injection attack patterns (obfuscated)."""
        graph_id = self._generate_id("injection")
        
        # Use pattern generator for security-safe patterns
        nodes = [
            {"id": self.patterns.get_sql_pattern(0), "type": "ComputeNode"},
            {"id": self.patterns.get_command_pattern(0), "type": "InputNode"},
            {"id": "normal_node", "type": self.patterns.get_xss_pattern(0)},
            {
                "id": "pattern_node",
                "type": "ComputeNode",
                "expression": self.patterns.get_template_pattern(0),
                "query": self.patterns.get_sql_pattern(1),
                "filter": self.patterns.get_command_pattern(1)
            },
            {
                "id": "encoded_patterns",
                "type": "Node",
                "data": base64.b64encode(self.patterns.get_xss_pattern(1).encode()).decode()
            }
        ]
        
        edges = [
            {"from": self.patterns.get_sql_pattern(0), "to": self.patterns.get_command_pattern(0)},
            {"from": "normal_node", "to": self.patterns.get_path_pattern(0)}
        ]
        
        return {
            "grammar_version": "1.0.0",
            "id": graph_id,
            "type": "Graph",
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "test_patterns": [
                    "sql_injection",
                    "command_injection",
                    "xss",
                    "path_traversal",
                    "template_injection"
                ]
            }
        }
    
    def _generate_deeply_nested_graph(self) -> Dict[str, Any]:
        """Generate a graph with extremely deep nesting (iterative to avoid recursion limit)."""
        graph_id = self._generate_id("deeply_nested")
        
        # Create deeply nested structure iteratively to avoid recursion limit
        def create_nested_structure_iterative(depth: int) -> Dict[str, Any]:
            result = {"value": "bottom", "depth": 0}
            for i in range(1, depth + 1):
                result = {
                    "level": i,
                    "nested": result
                }
            return result
        
        deep_structure = create_nested_structure_iterative(self.nesting_depth)
        
        nodes = [
            {
                "id": "nested_node",
                "type": "ComputeNode",
                "metadata": deep_structure
            }
        ]
        
        # Create nested array structure iteratively
        nested_arrays = []
        current = nested_arrays
        for i in range(min(50, self.nesting_depth)):
            new_array = []
            current.append(new_array)
            current = new_array
        current.append({"id": "deep_array_node", "type": "Node"})
        
        return {
            "grammar_version": "1.0.0",
            "id": graph_id,
            "type": "Graph",
            "nodes": nodes,
            "edges": [],
            "nested_arrays": nested_arrays,
            "deep_object": deep_structure,
            "metadata": {
                "nesting_depth": self.nesting_depth,
                "test_type": "deep_nesting"
            }
        }
    
    def _generate_null_fields_graph(self) -> Dict[str, Any]:
        """Generate a graph with null values in various fields."""
        graph_id = self._generate_id("null_fields")
        
        return {
            "grammar_version": None,
            "id": graph_id,
            "type": "Graph",
            "nodes": [
                None,
                {"id": None, "type": "ComputeNode"},
                {"id": "node1", "type": None},
                {"id": "node2", "type": "Node", "data": None},
                {"id": "node3", "type": "Node", "metadata": None, "attributes": None}
            ],
            "edges": [
                None,
                {"from": None, "to": None},
                {"from": "node1", "to": None},
                {"from": None, "to": "node2", "type": None}
            ],
            "metadata": None
        }
    
    def _generate_empty_graph(self) -> Dict[str, Any]:
        """Generate various forms of empty graphs (deterministic based on count)."""
        graph_id = self._generate_id("empty")
        
        # Use generated count for deterministic selection
        variant_index = self.generated_count % 4
        
        empty_variants = [
            {},  # Completely empty
            {"nodes": [], "edges": []},  # Empty arrays
            {"id": graph_id},  # Only ID
            {"grammar_version": "1.0.0", "id": "", "type": "", "nodes": [], "edges": []}  # Empty strings
        ]
        
        return empty_variants[variant_index]
    
    def _generate_unicode_exploit_graph(self) -> Dict[str, Any]:
        """Generate a graph with problematic Unicode characters."""
        graph_id = self._generate_id("unicode")
        
        # Various problematic Unicode scenarios (properly escaped)
        rtl_override = "\u202E"  # Right-to-left override
        zero_width = "\u200B"  # Zero-width space
        replacement = "\uFFFD"  # Replacement character
        combining_chars = "a\u0300\u0301\u0302\u0303\u0304"  # Multiple combining characters
        emoji_spam = "🔥" * 1000
        mixed_scripts = "ЛатинtекстÐ中文עברית"
        
        # Handle null byte specially - encode it for storage but mark for special handling
        null_byte_marker = "<NULL_BYTE>"
        
        nodes = [
            {"id": f"node{rtl_override}1", "type": "ComputeNode"},
            {"id": f"node{null_byte_marker}2", "type": "InputNode", "has_null_byte": True},
            {"id": f"{zero_width}invisible{zero_width}", "type": "OutputNode"},
            {"id": combining_chars * 10, "type": "Node"},
            {"id": emoji_spam, "type": "Node"},
            {"id": mixed_scripts, "type": "Node"},
            {
                "id": "unicode_categories",
                "type": "Node",
                "bidi": rtl_override,
                "zero_width": zero_width,
                "combining": combining_chars,
                "emoji": "🔥🔥🔥",
                "mixed": mixed_scripts
            }
        ]
        
        edges = [
            {"from": f"node{rtl_override}1", "to": f"{zero_width}invisible{zero_width}"}
        ]
        
        return {
            "grammar_version": "1.0.0",
            "id": graph_id,
            "type": "Graph",
            "nodes": nodes,
            "edges": edges,
            "description": f"Test{null_byte_marker}Graph{replacement}",
            "metadata": {
                "test_type": "unicode_exploit",
                "null_byte_marker": null_byte_marker
            }
        }
    
    def _generate_type_confusion_graph(self) -> Dict[str, Any]:
        """Generate a graph with intentional type confusion."""
        graph_id = self._generate_id("type_confusion")
        
        # Mix different representations testing parser type handling
        # Note: JSON serialization will normalize these, but the dict representation
        # tests parser's handling of mixed types before serialization
        return {
            "grammar_version": "1.0.0",
            "id": graph_id,
            "type": "Graph",
            "nodes": [
                {"id": "str_1", "type": "Node", "value": "1"},  # String "1"
                {"id": "num_1", "type": "Node", "value": 1},    # Number 1
                {"id": "str_true", "type": "Node", "value": "true"},  # String "true"
                {"id": "bool_true", "type": "Node", "value": True},   # Boolean true
                {"id": "str_null", "type": "Node", "value": "null"},  # String "null"
                {"id": "actual_null", "type": "Node", "value": None}, # Actual null
                {"id": "str_0", "type": "Node", "value": "0"},    # String "0"
                {"id": "num_0", "type": "Node", "value": 0},      # Number 0
                {"id": "bool_false", "type": "Node", "value": False}, # Boolean false
                {"id": "empty_str", "type": "Node", "value": ""},  # Empty string
                {"id": "str_obj", "type": "Node", "value": "{}"},  # String "{}"
                {"id": "actual_obj", "type": "Node", "value": {}}, # Actual empty object
            ],
            "edges": [
                {"from": "str_1", "to": "num_1"},
                {"from": "str_true", "to": "bool_true"},
                {"from": "str_0", "to": "bool_false"}
            ],
            "metadata": {
                "test_type": "type_confusion",
                "type_pairs": [
                    {"string": "1", "number": 1},
                    {"string": "true", "boolean": True},
                    {"string": "null", "null": None},
                    {"string": "0", "number": 0, "boolean": False}
                ]
            }
        }
    
    def _generate_oversized_graph(self) -> Dict[str, Any]:
        """Generate an extremely large graph to test size limits (configurable)."""
        graph_id = self._generate_id("oversized")
        
        nodes = []
        for i in range(self.oversized_nodes):
            nodes.append({
                "id": f"node_{i}",
                "type": random.choice(["InputNode", "OutputNode", "ComputeNode"]),
                "data": "x" * 1000,  # 1KB of data per node
                "metadata": {
                    "index": i,
                    "random": random.random(),
                    "description": f"This is node {i} with some padding text" * 10
                }
            })
        
        edges = []
        for i in range(self.oversized_edges):
            edges.append({
                "from": f"node_{random.randint(0, self.oversized_nodes-1)}",
                "to": f"node_{random.randint(0, self.oversized_nodes-1)}",
                "weight": random.random(),
                "type": random.choice(["data", "control", "dependency"])
            })
        
        return {
            "grammar_version": "1.0.0",
            "id": graph_id,
            "type": "Graph",
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "size": "oversized",
                "node_count": self.oversized_nodes,
                "edge_count": self.oversized_edges,
                "approximate_size_mb": (self.oversized_nodes * 2) / 1024
            }
        }
    
    def _generate_malformed_json_graph(self) -> Dict[str, Any]:
        """
        Generate a graph that tests special JSON values.
        Note: This creates valid Python dicts with special values that require
        custom JSON encoding. The save_graph() method handles serialization.
        """
        graph_id = self._generate_id("malformed")
        
        # Include special values that require custom JSON handling
        return {
            "grammar_version": "1.0.0",
            "id": graph_id,
            "type": "Graph",
            "nodes": [
                {
                    "id": "node_with_special_chars",
                    "type": "Node\nwith\nnewlines\tand\ttabs",
                    "data": "backslash\\test\\path",
                    "quotes": "test\"with'quotes",
                    "special_floats": {
                        "infinity": float('inf'),
                        "nan": float('nan'),
                        "negative_infinity": float('-inf')
                    }
                }
            ],
            "edges": [
                {
                    "from": "node\rwith\rcarriage\rreturn",
                    "to": "node\fwith\fform\ffeed"
                }
            ],
            "metadata": {
                "test_type": "special_json_values",
                "contains_special_floats": True
            }
        }
    
    def _generate_id(self, prefix: str = "graph") -> str:
        """Generate a unique ID for graphs."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
        return f"{prefix}_{timestamp}_{random_suffix}"
    
    def generate_batch(self, count: int = 10, shuffle: bool = True) -> List[Dict[str, Any]]:
        """
        Generate a batch of invalid graphs with various malicious patterns.
        Ensures at least one of each type is generated.
        
        Args:
            count: Number of graphs to generate
            shuffle: Whether to randomize the order of graph types
            
        Returns:
            List of generated graphs
        """
        graphs = []
        
        # Ensure we generate at least one of each type
        generation_methods = [
            self._generate_missing_fields_graph,
            self._generate_invalid_types_graph,
            self._generate_circular_dependency_graph,
            self._generate_duplicate_ids_graph,
            self._generate_invalid_references_graph,
            self._generate_buffer_overflow_graph,
            self._generate_injection_attempt_graph,
            self._generate_deeply_nested_graph,
            self._generate_null_fields_graph,
            self._generate_empty_graph,
            self._generate_unicode_exploit_graph,
            self._generate_type_confusion_graph,
            self._generate_oversized_graph,
            self._generate_malformed_json_graph
        ]
        
        # Generate at least one of each type if count allows
        for i, method in enumerate(generation_methods):
            if i < count:
                graphs.append(method())
        
        # Fill remaining with random selections
        while len(graphs) < count:
            method = random.choice(generation_methods)
            graphs.append(method())
        
        if shuffle:
            random.shuffle(graphs)
        
        return graphs
    
    def save_graph(self, graph: Dict[str, Any], output_dir: Optional[str] = None):
        """
        Save the graph to a JSON file with proper error handling.
        
        Args:
            graph: The graph to save
            output_dir: Directory to save to (uses self.output_dir if not specified)
        """
        if output_dir is None:
            output_dir = self.output_dir
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename based on graph ID or create one
        if isinstance(graph, dict) and "id" in graph:
            filename = f"{graph['id']}.json"
        else:
            filename = f"malformed_{self.generated_count}.json"
        
        file_path = os.path.join(output_dir, filename)
        
        try:
            # Custom JSON encoder for special float values
            def json_encoder(obj):
                if isinstance(obj, float):
                    if obj != obj:  # NaN
                        return "NaN"
                    elif obj == float('inf'):
                        return "Infinity"
                    elif obj == float('-inf'):
                        return "-Infinity"
                return obj
            
            # Use separate variable for error file to avoid scope confusion
            with open(file_path, "w", encoding='utf-8') as output_file:
                json.dump(graph, output_file, indent=2, ensure_ascii=False, default=json_encoder)
                
            if self.verbose:
                print(f"Saved graph to: {file_path}")
                
        except Exception as e:
            print(f"Error saving graph: {e}")
            # Save as .txt if JSON serialization fails - use different variable name
            error_file_path = file_path.replace('.json', '_error.txt')
            try:
                with open(error_file_path, "w", encoding='utf-8') as error_file:
                    error_file.write(f"Error: {e}\n")
                    error_file.write(f"Graph content:\n{str(graph)}")
                print(f"Saved error details to: {error_file_path}")
            except Exception as save_error:
                print(f"Failed to save error file: {save_error}")
    
    def save_batch(self, graphs: List[Dict[str, Any]], output_dir: Optional[str] = None):
        """
        Save a batch of graphs to files.
        
        Args:
            graphs: List of graphs to save
            output_dir: Directory to save to
        """
        for graph in graphs:
            self.save_graph(graph, output_dir)
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a report of the testing session.
        
        Returns:
            Dictionary containing test generation statistics
        """
        return {
            "seed": self.seed,
            "generated_count": self.generated_count,
            "output_directory": self.output_dir,
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "oversized_nodes": self.oversized_nodes,
                "oversized_edges": self.oversized_edges,
                "buffer_size_kb": self.buffer_size_kb,
                "nesting_depth": self.nesting_depth
            },
            "test_categories": [
                "missing_fields",
                "invalid_types",
                "circular_dependencies",
                "duplicate_ids",
                "invalid_references",
                "buffer_overflow",
                "injection_attempts",
                "deeply_nested",
                "null_fields",
                "empty_graphs",
                "unicode_exploits",
                "type_confusion",
                "oversized_graphs",
                "malformed_json"
            ],
            "security_note": "All injection patterns are obfuscated for safety"
        }
    
    def validate_generated_graphs(self, graphs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate that generated graphs actually violate expected constraints.
        
        Args:
            graphs: List of graphs to validate
            
        Returns:
            Validation report
        """
        validation_results = {
            "total_graphs": len(graphs),
            "violations_found": defaultdict(int),
            "graphs_by_type": defaultdict(int)
        }
        
        for graph in graphs:
            # Determine graph type from ID
            graph_id = graph.get("id", "unknown")
            if isinstance(graph_id, str):
                graph_type = graph_id.split("_")[0] if "_" in graph_id else "unknown"
                validation_results["graphs_by_type"][graph_type] += 1
            
            # Check for common violations
            if "grammar_version" not in graph or graph.get("grammar_version") is None:
                validation_results["violations_found"]["missing_grammar_version"] += 1
            
            if "id" not in graph or graph.get("id") is None:
                validation_results["violations_found"]["missing_id"] += 1
            
            if "type" not in graph or graph.get("type") is None:
                validation_results["violations_found"]["missing_type"] += 1
            
            if "nodes" not in graph:
                validation_results["violations_found"]["missing_nodes"] += 1
            elif not isinstance(graph.get("nodes"), list):
                validation_results["violations_found"]["nodes_not_array"] += 1
            
            if "edges" not in graph:
                validation_results["violations_found"]["missing_edges"] += 1
            elif not isinstance(graph.get("edges"), list):
                validation_results["violations_found"]["edges_not_array"] += 1
        
        return dict(validation_results)

def main():
    """
    Main function to generate a comprehensive test suite of malicious graphs.
    """
    # Initialize generator with verbose output
    generator = IRGenerator(verbose=True)
    
    # Configure generator
    generator.configure(
        oversized_nodes=1000,
        oversized_edges=2000,
        buffer_size_kb=100,
        nesting_depth=50
    )
    
    print(f"Starting malicious IR generation with seed: {generator.seed}")
    print("-" * 50)
    
    # Generate a comprehensive batch of test cases
    graphs = generator.generate_batch(count=20, shuffle=True)
    
    # Save all generated graphs
    generator.save_batch(graphs)
    
    # Validate generated graphs
    validation_report = generator.validate_generated_graphs(graphs)
    print("\nValidation Report:")
    print(f"  Total graphs: {validation_report['total_graphs']}")
    print(f"  Violations found: {dict(validation_report['violations_found'])}")
    print(f"  Graphs by type: {dict(validation_report['graphs_by_type'])}")
    
    # Save the generation report
    report = generator.generate_report()
    report["validation"] = validation_report
    
    report_path = os.path.join(generator.output_dir, "generation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print("-" * 50)
    print(f"Generation complete!")
    print(f"Generated {len(graphs)} malicious graphs")
    print(f"Saved to: {generator.output_dir}")
    print(f"Report saved to: {report_path}")
    print(f"Seed for reproducibility: {generator.seed}")
    print("\nSecurity Note: All injection patterns are obfuscated and safe for storage")

if __name__ == "__main__":
    main()