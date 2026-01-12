"""
Tests for Graph-Aware Evolution Engine

Tests integration of metaprogramming handlers with evolution engine.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch
from src.graph_aware_evolution import GraphAwareEvolutionEngine, create_graph_aware_engine


@pytest.fixture
def sample_graph():
    """Create a simple test graph."""
    return {
        "id": "test_graph",
        "nodes": [
            {"id": "n1", "type": "INPUT", "params": {"value": 1}},
            {"id": "n2", "type": "ADD", "params": {}},
            {"id": "n3", "type": "MULTIPLY", "params": {}},
            {"id": "n4", "type": "OUTPUT", "params": {}}
        ],
        "edges": [
            {"from": {"node": "n1"}, "to": {"node": "n2"}},
            {"from": {"node": "n2"}, "to": {"node": "n3"}},
            {"from": {"node": "n3"}, "to": {"node": "n4"}},
        ]
    }


@pytest.fixture
def mock_runtime_with_metaprog():
    """Create mock runtime with metaprogramming support."""
    runtime = Mock()
    
    # Mock NSO aligner (authorize by default for evolution)
    mock_aligner = Mock()
    mock_aligner.multi_model_audit = Mock(return_value="safe")
    
    mock_optimizer = Mock()
    mock_optimizer.nso = mock_aligner
    
    mock_extensions = Mock()
    mock_extensions.autonomous_optimizer = mock_optimizer
    
    runtime.extensions = mock_extensions
    runtime.execute_graph = AsyncMock(return_value=Mock(fitness=0.8))
    
    return runtime


class TestGraphAwareEvolutionEngine:
    """Test graph-aware evolution engine."""
    
    def test_init_without_runtime(self):
        """Test initialization without runtime (fallback mode)."""
        engine = GraphAwareEvolutionEngine(
            population_size=10,
            max_generations=5
        )
        
        assert engine.population_size == 10
        assert engine.max_generations == 5
        assert engine.metaprogramming_enabled == False
        assert engine.meta_stats["mutations_via_metaprog"] == 0
    
    def test_init_with_runtime(self, mock_runtime_with_metaprog, tmp_path):
        """Test initialization with runtime."""
        # Create a temporary mutator.json
        mutator_file = tmp_path / "mutator.json"
        mutator_file.write_text('{"id": "test_mutator", "nodes": [], "edges": []}')
        
        engine = GraphAwareEvolutionEngine(
            population_size=10,
            runtime=mock_runtime_with_metaprog,
            mutator_graph_path=str(mutator_file)
        )
        
        assert engine.runtime is not None
        assert engine.mutator_graph is not None
        assert engine.metaprogramming_enabled == True
    
    def test_fallback_mutation_no_runtime(self, sample_graph):
        """Test mutation falls back to dict manipulation without runtime."""
        engine = GraphAwareEvolutionEngine(population_size=5)
        
        mutated = engine._apply_random_mutation(sample_graph)
        
        # Should have mutated (or returned original if no valid mutation)
        assert isinstance(mutated, dict)
        assert "nodes" in mutated
        assert engine.meta_stats["mutations_via_dict"] >= 1
        assert engine.meta_stats["mutations_via_metaprog"] == 0
    
    @pytest.mark.asyncio
    async def test_metaprogramming_mutation(self, sample_graph, mock_runtime_with_metaprog):
        """Test mutation through metaprogramming pipeline."""
        engine = GraphAwareEvolutionEngine(
            population_size=5,
            runtime=mock_runtime_with_metaprog
        )
        engine.metaprogramming_enabled = True
        
        # Mock metaprogramming handlers to simulate successful mutation
        with patch('src.unified_runtime.metaprogramming_handlers.pattern_compile_node') as mock_compile, \
             patch('src.unified_runtime.metaprogramming_handlers.find_subgraph_node') as mock_find, \
             patch('src.unified_runtime.metaprogramming_handlers.graph_splice_node') as mock_splice, \
             patch('src.unified_runtime.metaprogramming_handlers.nso_modify_node') as mock_nso, \
             patch('src.unified_runtime.metaprogramming_handlers.ethical_label_node') as mock_ethical, \
             patch('src.unified_runtime.metaprogramming_handlers.graph_commit_node') as mock_commit:
            
            # Setup mock returns
            mock_compile.return_value = {"status": "success", "pattern_out": {"nodes": []}}
            mock_find.return_value = {"status": "success", "match_out": {"match_count": 1, "matches": [{"start_idx": 0, "end_idx": 0}]}}
            mock_splice.return_value = {"status": "success", "graph_out": sample_graph}
            mock_nso.return_value = {"nso_out": {"authorized": True}}
            mock_ethical.return_value = {"label_out": {"approved": True}}
            mock_commit.return_value = {
                "status": "success",
                "committed_graph": sample_graph,
                "version": {"hash": "abc123"}
            }
            
            mutated = await engine._apply_metaprogramming_mutation(sample_graph)
            
            assert mutated == sample_graph
            assert mock_compile.called
            assert mock_find.called
            assert mock_splice.called
            assert mock_nso.called
            assert mock_ethical.called
            assert mock_commit.called
    
    @pytest.mark.asyncio
    async def test_metaprogramming_nso_denial(self, sample_graph, mock_runtime_with_metaprog):
        """Test mutation blocked by NSO authorization."""
        engine = GraphAwareEvolutionEngine(
            population_size=5,
            runtime=mock_runtime_with_metaprog
        )
        engine.metaprogramming_enabled = True
        
        with patch('src.unified_runtime.metaprogramming_handlers.pattern_compile_node') as mock_compile, \
             patch('src.unified_runtime.metaprogramming_handlers.find_subgraph_node') as mock_find, \
             patch('src.unified_runtime.metaprogramming_handlers.graph_splice_node') as mock_splice, \
             patch('src.unified_runtime.metaprogramming_handlers.nso_modify_node') as mock_nso:
            
            mock_compile.return_value = {"status": "success", "pattern_out": {"nodes": []}}
            mock_find.return_value = {"status": "success", "match_out": {"match_count": 1}}
            mock_splice.return_value = {"status": "success", "graph_out": sample_graph}
            mock_nso.return_value = {"nso_out": {"authorized": False}}  # DENY
            
            mutated = await engine._apply_metaprogramming_mutation(sample_graph)
            
            # Should return original graph when denied
            assert mutated == sample_graph
            assert engine.meta_stats["authorization_denials"] == 1
    
    def test_generate_mutation_patterns(self, sample_graph):
        """Test pattern and template generation for mutations."""
        engine = GraphAwareEvolutionEngine(population_size=5)
        
        # Generate several patterns
        patterns_generated = set()
        for _ in range(20):
            pattern, template = engine._generate_mutation_pattern_and_template(sample_graph)
            
            assert isinstance(pattern, dict)
            assert isinstance(template, dict)
            assert "nodes" in pattern
            assert "nodes" in template
            
            # Record pattern types
            if pattern.get("nodes"):
                pattern_type = pattern["nodes"][0].get("type")
                patterns_generated.add(pattern_type)
        
        # Should generate diverse patterns
        assert len(patterns_generated) > 0
    
    def test_evolution_with_fallback(self, sample_graph):
        """Test full evolution cycle with fallback to dict mutation."""
        engine = GraphAwareEvolutionEngine(
            population_size=5,
            max_generations=3
        )
        
        # Seed with sample graph
        engine.initialize_population(seed_graph=sample_graph)  # Fixed: removed underscore
        
        # Simple fitness function
        def fitness_fn(graph):
            return len(graph.get("nodes", [])) / 10.0
        
        # Evolve
        best = engine.evolve(fitness_fn, generations=2)
        
        assert best is not None
        assert best.fitness >= 0
        assert engine.meta_stats["mutations_via_dict"] > 0
        assert engine.meta_stats["mutations_via_metaprog"] == 0
    
    def test_get_metaprogramming_stats(self):
        """Test metaprogramming statistics tracking."""
        engine = GraphAwareEvolutionEngine(population_size=5)
        
        # Simulate some mutations
        engine.meta_stats["mutations_via_metaprog"] = 8
        engine.meta_stats["mutations_via_dict"] = 2
        engine.meta_stats["authorization_denials"] = 1
        engine.meta_stats["ethical_blocks"] = 1
        
        stats = engine.get_metaprogramming_stats()
        
        assert stats["total_mutations"] == 10
        assert stats["metaprog_percentage"] == 80.0
        assert stats["safety_block_rate"] == 25.0  # 2 blocks out of 8 metaprog attempts
    
    def test_factory_function(self, mock_runtime_with_metaprog):
        """Test factory function creates engine correctly."""
        engine = create_graph_aware_engine(
            runtime=mock_runtime_with_metaprog,
            mutator_graph_path="graphs/mutator.json",
            population_size=15,
            max_generations=20
        )
        
        assert isinstance(engine, GraphAwareEvolutionEngine)
        assert engine.population_size == 15
        assert engine.max_generations == 20


class TestPatternTemplateGeneration:
    """Test pattern and template generation strategies."""
    
    def test_add_node_enhancement(self):
        """Test ADD node enhancement pattern."""
        engine = GraphAwareEvolutionEngine(population_size=5)
        graph = {"nodes": [{"id": "n1", "type": "ADD"}], "edges": []}
        
        # Generate patterns until we get ADD
        for _ in range(50):
            pattern, template = engine._generate_mutation_pattern_and_template(graph)
            
            if pattern.get("nodes") and pattern["nodes"][0].get("type") == "ADD":
                assert template["nodes"][0].get("type") == "ADD"
                assert "params" in template["nodes"][0]
                break
    
    def test_multiply_node_enhancement(self):
        """Test MULTIPLY node enhancement pattern."""
        engine = GraphAwareEvolutionEngine(population_size=5)
        graph = {"nodes": [{"id": "n1", "type": "MULTIPLY"}], "edges": []}
        
        # Generate patterns until we get MULTIPLY
        for _ in range(50):
            pattern, template = engine._generate_mutation_pattern_and_template(graph)
            
            if pattern.get("nodes") and pattern["nodes"][0].get("type") == "MULTIPLY":
                assert template["nodes"][0].get("type") == "MULTIPLY"
                break
    
    def test_fallback_pattern_empty_graph(self):
        """Test pattern generation with empty graph."""
        engine = GraphAwareEvolutionEngine(population_size=5)
        empty_graph = {"nodes": [], "edges": []}
        
        pattern, template = engine._generate_mutation_pattern_and_template(empty_graph)
        
        # Empty graph may get default ADD pattern or no-op pattern
        # Just check it returns valid structures
        assert isinstance(pattern, dict)
        assert isinstance(template, dict)


class TestIntegrationWithEvolutionEngine:
    """Test integration with base evolution engine."""
    
    def test_inherits_base_functionality(self, sample_graph):
        """Test that graph-aware engine inherits base functionality."""
        engine = GraphAwareEvolutionEngine(population_size=10)
        
        # Should have base evolution engine methods
        assert hasattr(engine, 'evolve')
        assert hasattr(engine, 'evolve_async')
        assert hasattr(engine, '_tournament_selection')
        assert hasattr(engine, '_crossover')
        
        # Should be able to use base mutations
        mutated = engine._mutate_add_node(sample_graph.copy())
        assert isinstance(mutated, dict)
    
    def test_cache_functionality(self):
        """Test that fitness cache still works."""
        engine = GraphAwareEvolutionEngine(population_size=5)
        
        # Cache should be initialized
        assert engine.fitness_cache is not None
        
        # Test cache operations
        engine.fitness_cache.put("key1", 0.8)
        assert engine.fitness_cache.get("key1") == 0.8
        
        stats = engine.fitness_cache.get_stats()
        assert stats["hits"] > 0


class TestSafetyIntegration:
    """Test safety system integration."""
    
    @pytest.mark.asyncio
    async def test_ethical_label_blocks_mutation(self, sample_graph, mock_runtime_with_metaprog):
        """Test ethical label can block mutations."""
        engine = GraphAwareEvolutionEngine(
            population_size=5,
            runtime=mock_runtime_with_metaprog
        )
        engine.metaprogramming_enabled = True
        
        with patch('src.unified_runtime.metaprogramming_handlers.pattern_compile_node') as mock_compile, \
             patch('src.unified_runtime.metaprogramming_handlers.find_subgraph_node') as mock_find, \
             patch('src.unified_runtime.metaprogramming_handlers.graph_splice_node') as mock_splice, \
             patch('src.unified_runtime.metaprogramming_handlers.nso_modify_node') as mock_nso, \
             patch('src.unified_runtime.metaprogramming_handlers.ethical_label_node') as mock_ethical:
            
            mock_compile.return_value = {"status": "success", "pattern_out": {"nodes": []}}
            mock_find.return_value = {"status": "success", "match_out": {"match_count": 1}}
            mock_splice.return_value = {"status": "success", "graph_out": sample_graph}
            mock_nso.return_value = {"nso_out": {"authorized": True}}
            mock_ethical.return_value = {"label_out": {"approved": False}}  # BLOCK
            
            mutated = await engine._apply_metaprogramming_mutation(sample_graph)
            
            assert mutated == sample_graph
            assert engine.meta_stats["ethical_blocks"] == 1
    
    @pytest.mark.asyncio
    async def test_no_match_returns_original(self, sample_graph, mock_runtime_with_metaprog):
        """Test that no pattern match returns original graph."""
        engine = GraphAwareEvolutionEngine(
            population_size=5,
            runtime=mock_runtime_with_metaprog
        )
        engine.metaprogramming_enabled = True
        
        with patch('src.unified_runtime.metaprogramming_handlers.pattern_compile_node') as mock_compile, \
             patch('src.unified_runtime.metaprogramming_handlers.find_subgraph_node') as mock_find:
            
            mock_compile.return_value = {"status": "success", "pattern_out": {"nodes": []}}
            mock_find.return_value = {"status": "no_match", "match_out": {"match_count": 0}}
            
            mutated = await engine._apply_metaprogramming_mutation(sample_graph)
            
            assert mutated == sample_graph


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
