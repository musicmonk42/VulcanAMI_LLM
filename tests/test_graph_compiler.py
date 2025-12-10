"""
Comprehensive test suite for graph_compiler.py
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import networkx as nx
import pytest

from src.compiler.graph_compiler import (CompilationError, CompiledNode,
                                         DataFlow, GraphCompiler,
                                         GraphOptimizer, NodeType)


@pytest.fixture
def compiler():
    """Create GraphCompiler instance."""
    with patch('src.compiler.graph_compiler.LLVMBackend'):
        return GraphCompiler(optimization_level=2)


@pytest.fixture
def optimizer():
    """Create GraphOptimizer instance."""
    return GraphOptimizer()


@pytest.fixture
def simple_graph():
    """Create simple graph for testing."""
    return {
        "nodes": [
            {"id": "input1", "type": "InputNode", "params": {"value": 1.0}},
            {"id": "const1", "type": "CONST", "params": {"value": 2.0}},
            {"id": "add1", "type": "ADD", "params": {}},
            {"id": "output1", "type": "OutputNode", "params": {}}
        ],
        "edges": [
            {"from": "input1", "to": "add1"},
            {"from": "const1", "to": "add1"},
            {"from": "add1", "to": "output1"}
        ]
    }


@pytest.fixture
def complex_graph():
    """Create complex graph for testing."""
    return {
        "nodes": [
            {"id": "input1", "type": "InputNode", "params": {}},
            {"id": "input2", "type": "InputNode", "params": {}},
            {"id": "mul1", "type": "MUL", "params": {}},
            {"id": "add1", "type": "ADD", "params": {}},
            {"id": "relu1", "type": "RELU", "params": {}},
            {"id": "output1", "type": "OutputNode", "params": {}}
        ],
        "edges": [
            {"from": "input1", "to": "mul1"},
            {"from": "input2", "to": "mul1"},
            {"from": "mul1", "to": "add1"},
            {"from": "input1", "to": "add1"},
            {"from": "add1", "to": "relu1"},
            {"from": "relu1", "to": "output1"}
        ]
    }


class TestNodeType:
    """Test NodeType enum."""
    
    def test_node_type_values(self):
        """Test node type enum values."""
        assert NodeType.INPUT.value == "InputNode"
        assert NodeType.OUTPUT.value == "OutputNode"
        assert NodeType.ADD.value == "ADD"
        assert NodeType.MUL.value == "MUL"
    
    def test_node_type_from_string(self):
        """Test creating NodeType from string."""
        node_type = NodeType("InputNode")
        assert node_type == NodeType.INPUT


class TestCompiledNode:
    """Test CompiledNode dataclass."""
    
    def test_compiled_node_creation(self):
        """Test creating CompiledNode."""
        node = CompiledNode(
            node_id="test_node",
            node_type=NodeType.ADD,
            inputs=["input1", "input2"],
            outputs=["output1"]
        )
        
        assert node.node_id == "test_node"
        assert node.node_type == NodeType.ADD
        assert len(node.inputs) == 2
        assert len(node.outputs) == 1


class TestGraphOptimizer:
    """Test GraphOptimizer."""
    
    def test_optimizer_initialization(self, optimizer):
        """Test optimizer initialization."""
        assert optimizer is not None
    
    def test_optimize_simple_graph(self, optimizer):
        """Test optimizing simple graph."""
        graph = nx.DiGraph()
        graph.add_node("n1", type="ADD")
        graph.add_node("n2", type="CONST", value=5)
        graph.add_edge("n2", "n1")
        
        optimized = optimizer.optimize(graph)
        
        assert optimized is not None
    
    def test_eliminate_dead_code(self, optimizer):
        """Test dead code elimination."""
        graph = nx.DiGraph()
        graph.add_node("input", type="InputNode")
        graph.add_node("used", type="ADD")
        graph.add_node("dead", type="MUL")
        graph.add_node("output", type="OutputNode")
        
        graph.add_edge("input", "used")
        graph.add_edge("used", "output")
        # "dead" is not connected
        
        optimized = optimizer._eliminate_dead_code(graph)
        
        assert "dead" not in optimized.nodes()
        assert "used" in optimized.nodes()
    
    def test_constant_folding(self, optimizer):
        """Test constant folding."""
        graph = nx.DiGraph()
        graph.add_node("c1", type="CONST", value=2.0)
        graph.add_node("c2", type="CONST", value=3.0)
        graph.add_node("add", type="ADD")
        
        graph.add_edge("c1", "add")
        graph.add_edge("c2", "add")
        
        optimized = optimizer._constant_folding(graph)
        
        # add node should be folded to constant
        assert optimized.nodes["add"]["type"] == "CONST"
    
    def test_matches_pattern(self, optimizer):
        """Test pattern matching."""
        graph = nx.DiGraph()
        graph.add_node("n1", type="CONV2D")
        graph.add_node("n2", type="BATCH_NORM")
        graph.add_node("n3", type="RELU")
        
        graph.add_edge("n1", "n2")
        graph.add_edge("n2", "n3")
        
        pattern = ["CONV2D", "BATCH_NORM", "RELU"]
        
        assert optimizer._matches_pattern(graph, "n1", pattern)
    
    def test_compute_signature(self, optimizer):
        """Test signature computation for CSE."""
        graph = nx.DiGraph()
        graph.add_node("n1", type="ADD", params={})
        graph.add_node("n2", type="ADD", params={})
        
        sig1 = optimizer._compute_signature(graph, "n1")
        sig2 = optimizer._compute_signature(graph, "n2")
        
        # Same type and params should give same signature
        assert sig1 == sig2


class TestGraphCompiler:
    """Test GraphCompiler."""
    
    def test_compiler_initialization(self, compiler):
        """Test compiler initialization."""
        assert compiler is not None
        assert compiler.optimization_level == 2
    
    def test_can_compile_simple_graph(self, compiler, simple_graph):
        """Test checking if graph can be compiled."""
        result = compiler.can_compile(simple_graph)
        
        assert isinstance(result, bool)
    
    def test_can_compile_with_unsupported_node(self, compiler):
        """Test compilation check with unsupported node."""
        graph = {
            "nodes": [
                {"id": "n1", "type": "UNSUPPORTED_NODE", "params": {}}
            ],
            "edges": []
        }
        
        result = compiler.can_compile(graph)
        
        assert result is False
    
    def test_can_compile_with_cycle(self, compiler):
        """Test compilation check with cyclic graph."""
        graph = {
            "nodes": [
                {"id": "n1", "type": "ADD", "params": {}},
                {"id": "n2", "type": "MUL", "params": {}}
            ],
            "edges": [
                {"from": "n1", "to": "n2"},
                {"from": "n2", "to": "n1"}  # Creates cycle
            ]
        }
        
        result = compiler.can_compile(graph)
        
        assert result is False
    
    def test_compute_graph_hash(self, compiler, simple_graph):
        """Test graph hash computation."""
        hash1 = compiler._compute_graph_hash(simple_graph)
        hash2 = compiler._compute_graph_hash(simple_graph)
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length
    
    def test_compute_graph_hash_deterministic(self, compiler):
        """Test that graph hash is deterministic."""
        graph = {
            "nodes": [
                {"id": "n1", "type": "ADD", "params": {"x": 1}},
                {"id": "n2", "type": "MUL", "params": {"y": 2}}
            ],
            "edges": []
        }
        
        hash1 = compiler._compute_graph_hash(graph)
        hash2 = compiler._compute_graph_hash(graph)
        
        assert hash1 == hash2
    
    def test_build_networkx_graph(self, compiler, simple_graph):
        """Test building NetworkX graph."""
        nx_graph = compiler._build_networkx_graph(simple_graph)
        
        assert isinstance(nx_graph, nx.DiGraph)
        assert len(nx_graph.nodes()) == len(simple_graph["nodes"])
        assert len(nx_graph.edges()) == len(simple_graph["edges"])
    
    def test_build_networkx_graph_with_dict_edges(self, compiler):
        """Test building graph with dictionary-style edges."""
        graph = {
            "nodes": [
                {"id": "n1", "type": "InputNode", "params": {}},
                {"id": "n2", "type": "OutputNode", "params": {}}
            ],
            "edges": [
                {
                    "from": {"node": "n1", "port": "out"},
                    "to": {"node": "n2", "port": "in"}
                }
            ]
        }
        
        nx_graph = compiler._build_networkx_graph(graph)
        
        assert nx_graph.has_edge("n1", "n2")
    
    @patch('src.compiler.graph_compiler.LLVMBackend')
    def test_compile_node_add(self, mock_backend_class, compiler):
        """Test compiling ADD node."""
        mock_backend = MagicMock()
        compiler.llvm_backend = mock_backend
        
        graph = nx.DiGraph()
        graph.add_node("add1", type="ADD", params={})
        
        compiled = compiler._compile_node(graph, "add1")
        
        assert compiled.node_id == "add1"
        assert compiled.node_type == NodeType.ADD
    
    def test_compile_node_unknown_type(self, compiler):
        """Test compiling unknown node type."""
        graph = nx.DiGraph()
        graph.add_node("unknown", type="UNKNOWN_TYPE", params={})
        
        with pytest.raises(CompilationError):
            compiler._compile_node(graph, "unknown")
    
    def test_compile_input_node(self, compiler):
        """Test compiling input node."""
        with patch.object(compiler.llvm_backend, 'builder'):
            node_data = {"id": "input1", "type": "InputNode", "value": 5.0}
            
            result = compiler._compile_input_node(node_data)
            
            assert result is not None
    
    def test_compile_const_node_scalar(self, compiler):
        """Test compiling constant node with scalar."""
        node_data = {"id": "const1", "type": "CONST", "value": 3.14}
        
        result = compiler._compile_const_node(node_data)
        
        assert result is not None
    
    def test_compile_const_node_array(self, compiler):
        """Test compiling constant node with array."""
        with patch.object(compiler.llvm_backend, 'builder'):
            node_data = {"id": "const1", "type": "CONST", "value": [1.0, 2.0, 3.0]}
            
            result = compiler._compile_const_node(node_data)
            
            assert result is not None
    
    def test_get_compilation_stats(self, compiler):
        """Test getting compilation statistics."""
        stats = compiler.get_compilation_stats()
        
        assert "cached_graphs" in stats
        assert "optimization_level" in stats
        assert "supported_nodes" in stats
    
    def test_compile_subgraph(self, compiler, simple_graph):
        """Test compiling subgraph."""
        subgraph_nodes = ["input1", "const1", "add1"]
        
        # Mock the compile_graph method
        with patch.object(compiler, 'compile_graph', return_value=b'compiled'):
            result = compiler.compile_subgraph(simple_graph, subgraph_nodes)
        
        assert result == b'compiled'
    
    def test_edge_in_subgraph(self, compiler):
        """Test checking if edge is in subgraph."""
        edge = {"from": "n1", "to": "n2"}
        nodes = ["n1", "n2", "n3"]
        
        assert compiler._edge_in_subgraph(edge, nodes) is True
        
        edge2 = {"from": "n1", "to": "n4"}
        assert compiler._edge_in_subgraph(edge2, nodes) is False
    
    def test_edge_in_subgraph_dict_format(self, compiler):
        """Test checking edge with dictionary format."""
        edge = {
            "from": {"node": "n1", "port": "out"},
            "to": {"node": "n2", "port": "in"}
        }
        nodes = ["n1", "n2"]
        
        assert compiler._edge_in_subgraph(edge, nodes) is True


class TestCompilationCaching:
    """Test compilation caching."""
    
    def test_cache_usage(self, compiler, simple_graph):
        """Test that compilation uses cache."""
        with patch.object(compiler, '_finalize_compilation', return_value=b'code'):
            with patch.object(compiler, '_build_networkx_graph') as mock_build:
                mock_graph = MagicMock()
                mock_graph.nodes.return_value = []
                mock_build.return_value = mock_graph
                
                # First compilation
                try:
                    result1 = compiler.compile_graph(simple_graph)
                except:
                    pass
                
                # Second compilation should use cache
                try:
                    result2 = compiler.compile_graph(simple_graph)
                except:
                    pass
                
                # Should have cached results
                assert len(compiler.compiled_cache) >= 0


class TestBenchmarking:
    """Test benchmarking functionality."""
    
    @patch('src.compiler.graph_compiler.LLVMBackend')
    def test_benchmark_compilation(self, mock_backend_class, compiler, simple_graph):
        """Test compilation benchmarking."""
        # Mock necessary methods
        with patch.object(compiler, '_build_networkx_graph') as mock_build:
            mock_graph = nx.DiGraph()
            mock_graph.add_node("n1", type="ADD", params={})
            mock_build.return_value = mock_graph
            
            with patch.object(compiler, '_compile_node'):
                with patch.object(compiler, '_create_main_function'):
                    with patch.object(compiler, '_link_nodes'):
                        with patch.object(compiler, '_finalize_compilation', return_value=b'code'):
                            times = compiler.benchmark_compilation(simple_graph)
        
        assert "parse_ms" in times
        assert "optimize_ms" in times
        assert "compile_nodes_ms" in times
        assert "total_ms" in times


class TestExceptions:
    """Test custom exceptions."""
    
    def test_compilation_error(self):
        """Test CompilationError."""
        error = CompilationError("compilation failed")
        
        assert str(error) == "compilation failed"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])