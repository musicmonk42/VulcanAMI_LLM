"""
Graph Compiler for Graphix IR
Compiles JSON graph representations to optimized native machine code
"""

import ctypes
import hashlib
import json
import logging
import os
import pickle
import struct
import subprocess
import tempfile
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import llvmlite.binding as llvm
import llvmlite.ir as ir
import networkx as nx
import numpy as np

from src.compiler.llvm_backend import CompiledFunction, DataType, LLVMBackend


class CompilationError(Exception):
    """Compilation-specific errors"""

    pass


class NodeType(Enum):
    """Supported node types for compilation"""

    INPUT = "InputNode"
    OUTPUT = "OutputNode"
    CONST = "CONST"
    ADD = "ADD"
    MUL = "MUL"
    MATRIX_MUL = "MATRIX_MUL"
    RELU = "RELU"
    SOFTMAX = "SOFTMAX"
    CONV2D = "CONV2D"
    BATCH_NORM = "BATCH_NORM"
    EMBEDDING = "EMBEDDING"
    ATTENTION = "ATTENTION"
    PHOTONIC_MVM = "PhotonicMVMNode"
    LOAD = "LOAD"
    STORE = "STORE"
    REDUCE_SUM = "REDUCE_SUM"
    REDUCE_MEAN = "REDUCE_MEAN"
    TRANSPOSE = "TRANSPOSE"
    RESHAPE = "RESHAPE"
    CONCAT = "CONCAT"
    SPLIT = "SPLIT"
    DYNAMIC_CODE = "DynamicCodeNode"
    GENERATIVE_AI = "GenerativeAINode"


@dataclass
class CompiledNode:
    """Represents a compiled node in the graph"""

    node_id: str
    node_type: NodeType
    llvm_function: Optional[ir.Function] = None
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    llvm_value: Optional[ir.Value] = None


@dataclass
class DataFlow:
    """Represents data flow between nodes"""

    source_node: str
    source_port: str
    target_node: str
    target_port: str
    data_type: DataType = DataType.FLOAT32
    shape: Optional[Tuple[int, ...]] = None


class GraphOptimizer:
    """Optimizes graph before compilation"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def optimize(self, graph: nx.DiGraph) -> nx.DiGraph:
        """Apply graph-level optimizations"""
        graph = self._fuse_operations(graph)
        graph = self._eliminate_dead_code(graph)
        graph = self._constant_folding(graph)
        graph = self._common_subexpression_elimination(graph)
        return graph

    def _fuse_operations(self, graph: nx.DiGraph) -> nx.DiGraph:
        """Fuse compatible operations for better performance"""
        # Identify fusable patterns
        patterns = [
            # Conv + BatchNorm + ReLU
            ["CONV2D", "BATCH_NORM", "RELU"],
            # MatMul + Add (Bias)
            ["MATRIX_MUL", "ADD"],
            # Multiple adds
            ["ADD", "ADD"],
        ]

        for pattern in patterns:
            # Find matching subgraphs
            for node in graph.nodes():
                if self._matches_pattern(graph, node, pattern):
                    self._fuse_pattern(graph, node, pattern)

        return graph

    def _matches_pattern(
        self, graph: nx.DiGraph, start_node: str, pattern: List[str]
    ) -> bool:
        """Check if subgraph matches pattern"""
        if len(pattern) == 0:
            return True

        node_data = graph.nodes[start_node]
        if node_data.get("type") != pattern[0]:
            return False

        if len(pattern) == 1:
            return True

        # Check successors
        successors = list(graph.successors(start_node))
        if len(successors) != 1:
            return False

        return self._matches_pattern(graph, successors[0], pattern[1:])

    def _fuse_pattern(self, graph: nx.DiGraph, start_node: str, pattern: List[str]):
        """Fuse matched pattern into single operation"""
        # Create fused node
        fused_id = f"fused_{start_node}"
        graph.add_node(fused_id, type=f"FUSED_{'_'.join(pattern)}", pattern=pattern)

        # Redirect edges
        for pred in graph.predecessors(start_node):
            graph.add_edge(pred, fused_id)

        # Find end of pattern and redirect outputs
        current = start_node
        for _ in range(len(pattern) - 1):
            current = list(graph.successors(current))[0]

        for succ in graph.successors(current):
            graph.add_edge(fused_id, succ)

        # Remove original nodes
        nodes_to_remove = []
        current = start_node
        for _ in range(len(pattern)):
            nodes_to_remove.append(current)
            successors = list(graph.successors(current))
            current = successors[0] if successors else None

        for node in nodes_to_remove:
            graph.remove_node(node)

    def _eliminate_dead_code(self, graph: nx.DiGraph) -> nx.DiGraph:
        """Remove unreachable nodes"""
        # Find all nodes reachable from inputs
        input_nodes = [
            n for n in graph.nodes() if graph.nodes[n].get("type") == "InputNode"
        ]
        reachable = set()

        for input_node in input_nodes:
            reachable.update(nx.descendants(graph, input_node))
            reachable.add(input_node)

        # Remove unreachable nodes
        unreachable = set(graph.nodes()) - reachable
        graph.remove_nodes_from(unreachable)

        return graph

    def _constant_folding(self, graph: nx.DiGraph) -> nx.DiGraph:
        """Evaluate constant expressions at compile time"""
        const_nodes = [
            n for n in graph.nodes() if graph.nodes[n].get("type") == "CONST"
        ]

        for node in const_nodes:
            # Check if all inputs are constants
            successors = list(graph.successors(node))
            for succ in successors:
                if self._can_fold(graph, succ):
                    self._fold_constants(graph, succ)

        return graph

    def _can_fold(self, graph: nx.DiGraph, node: str) -> bool:
        """Check if node can be constant folded"""
        node_type = graph.nodes[node].get("type")
        if node_type not in ["ADD", "MUL"]:
            return False

        # Check if all inputs are constants
        for pred in graph.predecessors(node):
            if graph.nodes[pred].get("type") != "CONST":
                return False

        return True

    def _fold_constants(self, graph: nx.DiGraph, node: str):
        """Fold constant expression"""
        node_type = graph.nodes[node].get("type")
        values = []

        for pred in graph.predecessors(node):
            values.append(graph.nodes[pred].get("value", 0))

        # Compute result
        if node_type == "ADD":
            result = sum(values)
        elif node_type == "MUL":
            result = np.prod(values)
        else:
            return

        # Replace with constant node
        graph.nodes[node]["type"] = "CONST"
        graph.nodes[node]["value"] = result

        # Remove input edges
        for pred in list(graph.predecessors(node)):
            graph.remove_edge(pred, node)

    def _common_subexpression_elimination(self, graph: nx.DiGraph) -> nx.DiGraph:
        """Eliminate duplicate computations"""
        # Hash nodes by their computation signature
        signatures = {}
        duplicates = defaultdict(list)

        for node in graph.nodes():
            sig = self._compute_signature(graph, node)
            if sig:
                if sig in signatures:
                    duplicates[sig].append(node)
                else:
                    signatures[sig] = node

        # Replace duplicates
        for sig, nodes in duplicates.items():
            original = signatures[sig]
            for dup in nodes:
                # Redirect all edges from dup to original
                for succ in list(graph.successors(dup)):
                    graph.add_edge(original, succ)
                graph.remove_node(dup)

        return graph

    def _compute_signature(self, graph: nx.DiGraph, node: str) -> Optional[str]:
        """Compute signature for CSE"""
        node_data = graph.nodes[node]
        node_type = node_data.get("type")

        if node_type in ["InputNode", "OutputNode"]:
            return None

        # Create signature from type and inputs
        sig_parts = [node_type]

        # Add sorted input signatures
        for pred in sorted(graph.predecessors(node)):
            pred_sig = self._compute_signature(graph, pred) or pred
            sig_parts.append(pred_sig)

        # Add parameters
        params = node_data.get("params", {})
        for key in sorted(params.keys()):
            sig_parts.append(f"{key}={params[key]}")

        return hashlib.sha256("|".join(sig_parts).encode()).hexdigest()


class GraphCompiler:
    """
    Main graph compiler that converts Graphix IR to native code
    """

    def __init__(self, optimization_level: int = 2):
        self.llvm_backend = LLVMBackend(optimization_level=optimization_level)
        self.optimization_level = optimization_level
        self.optimizer = GraphOptimizer()
        self.logger = logging.getLogger(__name__)
        self.compiled_cache = {}

        # Supported node types for compilation
        self.compilable_nodes = {
            NodeType.CONST,
            NodeType.ADD,
            NodeType.MUL,
            NodeType.MATRIX_MUL,
            NodeType.RELU,
            NodeType.SOFTMAX,
            NodeType.PHOTONIC_MVM,
            NodeType.LOAD,
            NodeType.STORE,
            NodeType.REDUCE_SUM,
            NodeType.REDUCE_MEAN,
            NodeType.TRANSPOSE,
            NodeType.RESHAPE,
        }

    def can_compile(self, graph: Dict[str, Any]) -> bool:
        """Check if graph can be compiled"""
        try:
            # Check all nodes are compilable
            for node in graph.get("nodes", []):
                node_type_str = node.get("type", "")
                try:
                    node_type = NodeType(node_type_str)
                    if node_type not in self.compilable_nodes and node_type not in {
                        NodeType.INPUT,
                        NodeType.OUTPUT,
                    }:
                        self.logger.debug(f"Node type {node_type_str} not compilable")
                        return False
                except ValueError:
                    self.logger.debug(f"Unknown node type: {node_type_str}")
                    return False

            # Check graph structure is valid
            nx_graph = self._build_networkx_graph(graph)
            if not nx.is_directed_acyclic_graph(nx_graph):
                self.logger.debug("Graph contains cycles")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error checking compilability: {e}")
            return False

    def compile_graph(self, graph: Dict[str, Any]) -> bytes:
        """
        Compile graph to native machine code
        """
        graph_hash = self._compute_graph_hash(graph)

        # Check cache
        if graph_hash in self.compiled_cache:
            self.logger.debug(f"Using cached compilation for {graph_hash}")
            return self.compiled_cache[graph_hash]

        try:
            # Build and optimize graph
            nx_graph = self._build_networkx_graph(graph)
            nx_graph = self.optimizer.optimize(nx_graph)

            # Topological sort for execution order
            exec_order = list(nx.topological_sort(nx_graph))

            # Create main function
            main_func = self._create_main_function(nx_graph, exec_order)

            # Compile nodes
            compiled_nodes = {}
            for node_id in exec_order:
                compiled_nodes[node_id] = self._compile_node(nx_graph, node_id)

            # Link nodes according to data flow
            self._link_nodes(nx_graph, compiled_nodes, main_func)

            # Optimize and compile to machine code
            machine_code = self._finalize_compilation()

            # Cache result
            self.compiled_cache[graph_hash] = machine_code

            return machine_code

        except Exception as e:
            raise CompilationError(f"Failed to compile graph: {e}")

    def _compute_graph_hash(self, graph: Dict[str, Any]) -> str:
        """Compute deterministic hash for graph"""
        graph_str = json.dumps(graph, sort_keys=True)
        return hashlib.sha256(graph_str.encode()).hexdigest()

    def _build_networkx_graph(self, graph: Dict[str, Any]) -> nx.DiGraph:
        """Convert JSON graph to NetworkX representation"""
        nx_graph = nx.DiGraph()

        # Add nodes
        for node in graph.get("nodes", []):
            nx_graph.add_node(
                node["id"],
                type=node.get("type"),
                params=node.get("params", {}),
                value=node.get("value"),
            )

        # Add edges
        for edge in graph.get("edges", []):
            from_data = edge.get("from")
            to_data = edge.get("to")

            # Handle different edge formats
            if isinstance(from_data, dict):
                source = from_data.get("node")
                port_from = from_data.get("port")
            else:
                source = from_data
                port_from = None

            if isinstance(to_data, dict):
                target = to_data.get("node")
                port_to = to_data.get("port")
            else:
                target = to_data
                port_to = None

            if source and target:
                nx_graph.add_edge(
                    source,
                    target,
                    type=edge.get("type", "data"),
                    port_from=port_from,
                    port_to=port_to,
                )

        return nx_graph

    def _create_main_function(
        self, graph: nx.DiGraph, exec_order: List[str]
    ) -> ir.Function:
        """Create main function for graph execution"""
        # Function signature: int graphix_main(void* inputs, void* outputs, int num_nodes)
        ptr_type = ir.IntType(8).as_pointer()
        func_type = ir.FunctionType(
            ir.IntType(32),  # return type
            [ptr_type, ptr_type, ir.IntType(32)],  # parameters
        )

        main_func = ir.Function(
            self.llvm_backend.module, func_type, name="graphix_main"
        )
        main_func.args[0].name = "inputs"
        main_func.args[1].name = "outputs"
        main_func.args[2].name = "num_nodes"

        # Create entry block
        entry_block = main_func.append_basic_block(name="entry")
        self.llvm_backend.builder = ir.IRBuilder(entry_block)

        return main_func

    def _compile_node(self, graph: nx.DiGraph, node_id: str) -> CompiledNode:
        """Compile individual node"""
        node_data = graph.nodes[node_id]
        node_type_str = node_data.get("type", "")

        try:
            node_type = NodeType(node_type_str)
        except ValueError:
            raise CompilationError(f"Unknown node type: {node_type_str}")

        compiled = CompiledNode(
            node_id=node_id, node_type=node_type, params=node_data.get("params", {})
        )

        # Get inputs and outputs
        compiled.inputs = list(graph.predecessors(node_id))
        compiled.outputs = list(graph.successors(node_id))

        # Handle special node types
        if node_type == NodeType.INPUT:
            compiled.llvm_value = self._compile_input_node(node_data)
        elif node_type == NodeType.OUTPUT:
            # Output nodes are handled in linking phase
            pass
        elif node_type == NodeType.CONST:
            compiled.llvm_value = self._compile_const_node(node_data)
        else:
            # Compile using backend
            backend_type = node_type.name
            compiled_func = self.llvm_backend.compile_node(
                backend_type, compiled.params
            )
            compiled.llvm_function = compiled_func.llvm_func

        return compiled

    def _compile_input_node(self, node_data: Dict) -> ir.Value:
        """Compile input node"""
        builder = self.llvm_backend.builder

        # Allocate space for input value
        value_type = ir.DoubleType()
        alloca = builder.alloca(value_type, name=f"input_{node_data.get('id')}")

        # Load initial value if provided
        if "value" in node_data:
            initial = ir.Constant(value_type, float(node_data["value"]))
            builder.store(initial, alloca)

        return alloca

    def _compile_const_node(self, node_data: Dict) -> ir.Value:
        """Compile constant node"""
        value = node_data.get("value", 0.0)

        if isinstance(value, (int, float)):
            return ir.Constant(ir.DoubleType(), float(value))
        elif isinstance(value, list):
            # Array constant
            array_type = ir.ArrayType(ir.FloatType(), len(value))
            array_const = ir.Constant(array_type, value)

            # Allocate and initialize
            builder = self.llvm_backend.builder
            alloca = builder.alloca(array_type)
            builder.store(array_const, alloca)
            return alloca
        else:
            raise CompilationError(f"Unsupported constant type: {type(value)}")

    def _link_nodes(
        self,
        graph: nx.DiGraph,
        compiled_nodes: Dict[str, CompiledNode],
        main_func: ir.Function,
    ):
        """Link compiled nodes according to data flow"""
        builder = self.llvm_backend.builder

        # Create value map for data flow
        value_map = {}

        # Process nodes in topological order
        for node_id in nx.topological_sort(graph):
            node = compiled_nodes[node_id]

            if node.node_type == NodeType.INPUT:
                # Input nodes already have values
                value_map[node_id] = builder.load(node.llvm_value)

            elif node.node_type == NodeType.CONST:
                # Constants have direct values
                value_map[node_id] = node.llvm_value

            elif node.node_type == NodeType.OUTPUT:
                # Store input value to output
                if node.inputs:
                    input_value = value_map.get(node.inputs[0])
                    if input_value:
                        # Store to output buffer
                        output_ptr = builder.gep(
                            main_func.args[1], [ir.Constant(ir.IntType(32), 0)]
                        )
                        output_typed = builder.bitcast(
                            output_ptr, ir.DoubleType().as_pointer()
                        )
                        builder.store(input_value, output_typed)

            elif node.llvm_function:
                # Call compiled function with inputs
                args = []
                for input_id in node.inputs:
                    input_val = value_map.get(input_id)
                    if input_val:
                        args.append(input_val)

                if args:
                    # Call function based on type
                    if node.node_type in [NodeType.ADD, NodeType.MUL]:
                        # Binary operations
                        if len(args) >= 2:
                            result = builder.call(node.llvm_function, args[:2])
                            value_map[node_id] = result
                    elif node.node_type == NodeType.RELU:
                        # Unary operations
                        if len(args) >= 1:
                            result = builder.call(node.llvm_function, [args[0]])
                            value_map[node_id] = result
                    elif node.node_type == NodeType.MATRIX_MUL:
                        # Matrix operations need special handling
                        self._link_matrix_mul(node, args, value_map, builder)
                    elif node.node_type == NodeType.PHOTONIC_MVM:
                        # Photonic operations
                        self._link_photonic_mvm(node, args, value_map, builder)

        # Return success
        builder.ret(ir.Constant(ir.IntType(32), 0))

    def _link_matrix_mul(
        self,
        node: CompiledNode,
        args: List[ir.Value],
        value_map: Dict,
        builder: ir.IRBuilder,
    ):
        """Link matrix multiplication node"""
        # Allocate output buffer
        output_size = node.params.get("output_size", (10, 10))
        output_type = ir.ArrayType(ir.FloatType(), output_size[0] * output_size[1])
        output_alloca = builder.alloca(output_type)

        # Call matrix multiplication
        if len(args) >= 2:
            # Convert arguments to pointers if needed
            mat_a_ptr = args[0]
            mat_b_ptr = args[1]

            # Get dimensions from params
            m = ir.Constant(ir.IntType(32), node.params.get("m", 10))
            n = ir.Constant(ir.IntType(32), node.params.get("n", 10))
            k = ir.Constant(ir.IntType(32), node.params.get("k", 10))

            # Call function
            builder.call(
                node.llvm_function, [mat_a_ptr, mat_b_ptr, output_alloca, m, n, k]
            )

            # Store result
            value_map[node.node_id] = output_alloca

    def _link_photonic_mvm(
        self,
        node: CompiledNode,
        args: List[ir.Value],
        value_map: Dict,
        builder: ir.IRBuilder,
    ):
        """Link photonic matrix-vector multiplication"""
        # Similar to matrix mul but with noise simulation
        output_size = node.params.get("output_size", 10)
        output_type = ir.ArrayType(ir.FloatType(), output_size)
        output_alloca = builder.alloca(output_type)

        if len(args) >= 2:
            matrix_ptr = args[0]
            vector_ptr = args[1]

            rows = ir.Constant(ir.IntType(32), node.params.get("rows", 10))
            cols = ir.Constant(ir.IntType(32), node.params.get("cols", 10))

            builder.call(
                node.llvm_function, [matrix_ptr, vector_ptr, output_alloca, rows, cols]
            )
            value_map[node.node_id] = output_alloca

    def _finalize_compilation(self) -> bytes:
        """Finalize compilation and generate machine code"""
        # Optimize module
        optimized = self.llvm_backend.optimize_module()

        # Generate object code
        target = llvm.Target.from_default_triple()
        target_machine = target.create_target_machine(
            opt=self.llvm_backend.optimization_level,
            codemodel="small",
            features="",
        )

        # Compile to machine code
        machine_code = target_machine.emit_object(optimized)

        # Create shared library
        with tempfile.NamedTemporaryFile(suffix=".o", delete=False) as f:
            f.write(machine_code)
            obj_file = f.name

        try:
            # Link to shared library
            so_file = tempfile.mktemp(suffix=".so")
            subprocess.run(
                ["gcc", "-shared", "-fPIC", obj_file, "-o", so_file],
                check=True,
                capture_output=True,
            )

            # Read compiled library
            with open(so_file, "rb") as f:
                library_bytes = f.read()

            # Cleanup
            os.unlink(so_file)

            return library_bytes

        finally:
            os.unlink(obj_file)

    def compile_subgraph(
        self, graph: Dict[str, Any], subgraph_nodes: List[str]
    ) -> bytes:
        """Compile a subgraph for fusion"""
        # Extract subgraph
        subgraph = {
            "nodes": [n for n in graph["nodes"] if n["id"] in subgraph_nodes],
            "edges": [
                e for e in graph["edges"] if self._edge_in_subgraph(e, subgraph_nodes)
            ],
        }

        # Compile subgraph
        return self.compile_graph(subgraph)

    def _edge_in_subgraph(self, edge: Dict, nodes: List[str]) -> bool:
        """Check if edge is within subgraph"""
        source = edge.get("from")
        target = edge.get("to")

        if isinstance(source, dict):
            source = source.get("node")
        if isinstance(target, dict):
            target = target.get("node")

        return source in nodes and target in nodes

    def get_compilation_stats(self) -> Dict[str, Any]:
        """Get compilation statistics"""
        return {
            "cached_graphs": len(self.compiled_cache),
            "total_cache_size": sum(len(code) for code in self.compiled_cache.values()),
            "optimization_level": self.llvm_backend.optimization_level,
            "supported_nodes": [t.value for t in self.compilable_nodes],
        }

    def benchmark_compilation(self, graph: Dict[str, Any]) -> Dict[str, float]:
        """Benchmark compilation phases"""
        import time

        times = {}

        # Parse phase
        start = time.perf_counter()
        nx_graph = self._build_networkx_graph(graph)
        times["parse_ms"] = (time.perf_counter() - start) * 1000

        # Optimization phase
        start = time.perf_counter()
        optimized = self.optimizer.optimize(nx_graph)
        times["optimize_ms"] = (time.perf_counter() - start) * 1000

        # Compilation phase
        start = time.perf_counter()
        exec_order = list(nx.topological_sort(optimized))
        compiled_nodes = {}
        for node_id in exec_order:
            compiled_nodes[node_id] = self._compile_node(optimized, node_id)
        times["compile_nodes_ms"] = (time.perf_counter() - start) * 1000

        # Linking phase
        start = time.perf_counter()
        main_func = self._create_main_function(optimized, exec_order)
        self._link_nodes(optimized, compiled_nodes, main_func)
        times["link_ms"] = (time.perf_counter() - start) * 1000

        # Code generation phase
        start = time.perf_counter()
        machine_code = self._finalize_compilation()
        times["codegen_ms"] = (time.perf_counter() - start) * 1000

        times["total_ms"] = sum(times.values())
        times["code_size_bytes"] = len(machine_code)

        return times
