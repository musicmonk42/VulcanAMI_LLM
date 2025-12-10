"""
Comprehensive test suite for evolution_engine.py
"""

import asyncio
import copy
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from evolution_engine import (MAX_CACHE_SIZE, CacheStatistics, EvolutionEngine,
                              Individual, LRUCache)


@pytest.fixture
def engine():
    """Create evolution engine."""
    return EvolutionEngine(
        population_size=20,
        mutation_rate=0.1,
        crossover_rate=0.7,
        max_generations=10,
        cache_size=100
    )


@pytest.fixture
def simple_graph():
    """Create simple test graph."""
    return {
        'grammar_version': '2.1.0',
        'nodes': [
            {'id': 'input', 'type': 'input', 'params': {}},
            {'id': 'n1', 'type': 'transform', 'params': {'value': 0.5}},
            {'id': 'output', 'type': 'output', 'params': {}}
        ],
        'edges': [
            {'source': 'input', 'target': 'n1', 'weight': 1.0},
            {'source': 'n1', 'target': 'output', 'weight': 1.0}
        ],
        'metadata': {}
    }


@pytest.fixture
def fitness_function():
    """Create simple fitness function."""
    def fitness(graph):
        return len(graph.get('nodes', [])) / 10.0
    return fitness


class TestIndividual:
    """Test Individual dataclass."""

    def test_initialization(self, simple_graph):
        """Test individual initialization."""
        ind = Individual(graph=simple_graph, fitness=0.5)

        assert ind.graph == simple_graph
        assert ind.fitness == 0.5
        assert ind.generation == 0
        assert len(ind.id) > 0

    def test_mutations_tracking(self, simple_graph):
        """Test mutation tracking."""
        ind = Individual(graph=simple_graph)
        ind.mutations.append("test_mutation")

        assert "test_mutation" in ind.mutations


class TestLRUCache:
    """Test LRU cache."""

    def test_initialization(self):
        """Test cache initialization."""
        cache = LRUCache(max_size=10)

        assert cache.max_size == 10
        assert cache.stats.current_size == 0

    def test_get_miss(self):
        """Test cache miss."""
        cache = LRUCache()

        value = cache.get("nonexistent")

        assert value is None
        assert cache.stats.misses == 1

    def test_put_and_get(self):
        """Test putting and getting."""
        cache = LRUCache()

        cache.put("key1", 0.5)
        value = cache.get("key1")

        assert value == 0.5
        assert cache.stats.hits == 1

    def test_lru_eviction(self):
        """Test LRU eviction."""
        cache = LRUCache(max_size=3)

        cache.put("key1", 1.0)
        cache.put("key2", 2.0)
        cache.put("key3", 3.0)
        cache.put("key4", 4.0)  # Should evict key1

        assert cache.get("key1") is None
        assert cache.get("key4") == 4.0
        assert cache.stats.evictions == 1

    def test_clear(self):
        """Test clearing cache."""
        cache = LRUCache()

        cache.put("key1", 1.0)
        cache.put("key2", 2.0)

        cache.clear()

        assert cache.stats.current_size == 0
        assert cache.get("key1") is None

    def test_get_stats(self):
        """Test getting statistics."""
        cache = LRUCache()

        cache.put("key1", 1.0)
        cache.get("key1")
        cache.get("key2")

        stats = cache.get_stats()

        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['hit_rate'] == 0.5


class TestEvolutionEngine:
    """Test EvolutionEngine class."""

    def test_initialization(self):
        """Test engine initialization."""
        engine = EvolutionEngine(population_size=50)

        assert engine.population_size == 50
        assert engine.generation == 0
        assert len(engine.population) == 0

    def test_parameter_validation(self):
        """Test parameter validation and capping."""
        engine = EvolutionEngine(
            population_size=2000,  # Over max
            mutation_rate=2.0,     # Over max
            tournament_size=50     # Over max
        )

        assert engine.population_size <= 1000
        assert engine.mutation_rate <= 1.0
        assert engine.tournament_size <= 20

    def test_sanitize_string(self, engine):
        """Test string sanitization."""
        dirty = "test;value`with$bad<chars>"
        clean = engine._sanitize_string(dirty)

        assert ';' not in clean
        assert '`' not in clean
        assert '$' not in clean

    def test_sanitize_params(self, engine):
        """Test parameter sanitization."""
        params = {
            'value': 0.5,
            'name': 'test;value',
            'count': 999999999,
            'items': list(range(20),  # Too many)
            'nested': {'key': 'value'}
        }

        safe = engine._sanitize_params(params)

        assert ';' not in safe.get('name', '')
        assert safe['count'] <= 1000000
        assert len(safe['items']) <= 10

    def test_validate_node(self, engine):
        """Test node validation."""
        valid_node = {'id': 'n1', 'type': 'transform', 'params': {}}

        assert engine._validate_node(valid_node)

    def test_validate_node_invalid_type(self, engine):
        """Test validating node with invalid type."""
        invalid_node = {'id': 'n1', 'type': 'malicious', 'params': {}}

        assert not engine._validate_node(invalid_node)

    def test_validate_edge(self, engine):
        """Test edge validation."""
        node_ids = {'n1', 'n2'}
        valid_edge = {'source': 'n1', 'target': 'n2', 'weight': 0.5}

        assert engine._validate_edge(valid_edge, node_ids)

    def test_validate_edge_nonexistent_nodes(self, engine):
        """Test validating edge with nonexistent nodes."""
        node_ids = {'n1'}
        invalid_edge = {'source': 'n1', 'target': 'n999'}

        assert not engine._validate_edge(invalid_edge, node_ids)

    def test_validate_graph(self, engine, simple_graph):
        """Test graph validation."""
        validated = engine._validate_graph(simple_graph)

        assert 'nodes' in validated
        assert 'edges' in validated
        assert len(validated['nodes']) > 0

    def test_validate_graph_empty(self, engine):
        """Test validating empty graph."""
        empty = {}
        validated = engine._validate_graph(empty)

        # Should return minimal graph
        assert len(validated['nodes']) >= 2

    def test_generate_minimal_graph(self, engine):
        """Test minimal graph generation."""
        graph = engine._generate_minimal_graph()

        assert len(graph['nodes']) == 2
        assert graph['nodes'][0]['type'] == 'input'
        assert graph['nodes'][1]['type'] == 'output'

    def test_initialize_population_random(self, engine):
        """Test random population initialization."""
        engine.initialize_population()

        assert len(engine.population) == engine.population_size
        assert all(isinstance(ind, Individual) for ind in engine.population)

    def test_initialize_population_seeded(self, engine, simple_graph):
        """Test seeded population initialization."""
        engine.initialize_population(seed_graph=simple_graph)

        assert len(engine.population) == engine.population_size
        # First individual should match seed
        assert engine.population[0].graph['nodes'][0]['id'] == 'input'

    def test_evolve_synchronous(self, engine):
        """Test synchronous evolution."""
        # Use a fitness function that won't easily reach 0.99 to test full evolution
        def fitness_fn(graph):
            # Scale down to ensure we don't trigger early termination
            return min(0.5, len(graph.get('nodes', [])) / 20.0)

        engine.initialize_population()

        best = engine.evolve(fitness_fn, generations=5)

        assert best is not None
        assert best.fitness >= 0
        assert engine.generation == 5

    @pytest.mark.asyncio
    async def test_evolve_async(self, engine, fitness_function):
        """Test asynchronous evolution."""
        engine.initialize_population()

        best = await engine.evolve_async(fitness_function, generations=3, max_workers=2)

        assert best is not None
        assert best.fitness >= 0

    def test_tournament_selection(self, engine):
        """Test tournament selection."""
        engine.initialize_population()

        # Set some fitnesses
        for i, ind in enumerate(engine.population):
            ind.fitness = i / len(engine.population)

        selected = engine._tournament_selection()

        assert selected in engine.population

    def test_crossover_single_point(self, engine, simple_graph):
        """Test single-point crossover."""
        graph1 = copy.deepcopy(simple_graph)
        graph2 = copy.deepcopy(simple_graph)
        graph2['nodes'].append({'id': 'extra', 'type': 'transform'})

        child1, child2 = engine._crossover_single_point(graph1, graph2)

        assert 'nodes' in child1
        assert 'nodes' in child2

    def test_crossover_uniform(self, engine, simple_graph):
        """Test uniform crossover."""
        graph1 = copy.deepcopy(simple_graph)
        graph2 = copy.deepcopy(simple_graph)

        child1, child2 = engine._crossover_uniform(graph1, graph2)

        assert 'nodes' in child1
        assert 'nodes' in child2

    def test_crossover_subgraph(self, engine):
        """Test subgraph crossover."""
        graph1 = {
            'nodes': [
                {'id': 'input', 'type': 'input'},
                {'id': 'n1', 'type': 'transform'},
                {'id': 'output', 'type': 'output'}
            ],
            'edges': [
                {'source': 'input', 'target': 'n1'},
                {'source': 'n1', 'target': 'output'}
            ]
        }

        graph2 = {
            'nodes': [
                {'id': 'input', 'type': 'input'},
                {'id': 'x1', 'type': 'filter'},
                {'id': 'output', 'type': 'output'}
            ],
            'edges': [
                {'source': 'input', 'target': 'x1'},
                {'source': 'x1', 'target': 'output'}
            ]
        }

        child1, child2 = engine._crossover_subgraph(graph1, graph2)

        assert 'nodes' in child1
        assert 'nodes' in child2

    def test_find_connected_subgraphs(self, engine, simple_graph):
        """Test finding connected subgraphs."""
        subgraphs = engine._find_connected_subgraphs(simple_graph)

        assert isinstance(subgraphs, list)

    def test_mutate_add_node(self, engine, simple_graph):
        """Test add node mutation."""
        original_count = len(simple_graph['nodes'])

        mutated = engine._mutate_add_node(copy.deepcopy(simple_graph))

        # May add node if under limit
        assert len(mutated['nodes']) >= original_count

    def test_mutate_remove_node(self, engine, simple_graph):
        """Test remove node mutation."""
        graph_with_extra = copy.deepcopy(simple_graph)
        graph_with_extra['nodes'].append({'id': 'extra', 'type': 'transform'})

        mutated = engine._mutate_remove_node(graph_with_extra)

        # Should maintain at least 2 nodes
        assert len(mutated['nodes']) >= 2

    def test_mutate_modify_edge(self, engine, simple_graph):
        """Test modify edge mutation."""
        original_weight = simple_graph['edges'][0].get('weight', 1.0)

        mutated = engine._mutate_modify_edge(copy.deepcopy(simple_graph))

        # Weight may have changed
        assert 'weight' in mutated['edges'][0]

    def test_mutate_change_parameter(self, engine, simple_graph):
        """Test change parameter mutation."""
        mutated = engine._mutate_change_parameter(copy.deepcopy(simple_graph))

        # Some node should have params
        assert any('params' in node for node in mutated['nodes'])

    def test_mutate_swap_nodes(self, engine):
        """Test swap nodes mutation."""
        graph = {
            'nodes': [
                {'id': 'input', 'type': 'input'},
                {'id': 'n1', 'type': 'transform'},
                {'id': 'n2', 'type': 'filter'},
                {'id': 'output', 'type': 'output'}
            ],
            'edges': []
        }

        mutated = engine._mutate_swap_nodes(copy.deepcopy(graph))

        # Nodes should still exist
        assert len(mutated['nodes']) == 4

    def test_generate_random_graph(self, engine):
        """Test random graph generation."""
        graph = engine._generate_random_graph()

        assert 'nodes' in graph
        assert 'edges' in graph
        assert len(graph['nodes']) >= 2

    def test_hash_graph(self, engine, simple_graph):
        """Test graph hashing."""
        hash1 = engine._hash_graph(simple_graph)
        hash2 = engine._hash_graph(simple_graph)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hash length

    def test_calculate_diversity(self, engine):
        """Test diversity calculation."""
        engine.initialize_population()

        # Set varied fitnesses
        for i, ind in enumerate(engine.population):
            ind.fitness = i / len(engine.population)

        diversity = engine._calculate_diversity()

        assert 0.0 <= diversity <= 1.0

    def test_inject_diversity(self, engine):
        """Test diversity injection."""
        engine.initialize_population()

        # Make all graphs the same
        same_graph = engine.population[0].graph
        for ind in engine.population:
            ind.graph = copy.deepcopy(same_graph)

        diversity_before = engine._calculate_diversity()

        engine._inject_diversity(fraction=0.3)

        diversity_after = engine._calculate_diversity()

        assert diversity_after >= diversity_before

    def test_clear_cache(self, engine):
        """Test cache clearing."""
        engine.fitness_cache.put("key1", 0.5)

        engine.clear_cache()

        assert engine.fitness_cache.get("key1") is None

    def test_get_cache_stats(self, engine):
        """Test getting cache stats."""
        stats = engine.get_cache_stats()

        assert 'hits' in stats
        assert 'misses' in stats
        assert 'current_size' in stats

    def test_get_statistics(self, engine):
        """Test getting statistics."""
        engine.initialize_population()

        stats = engine.get_statistics()

        assert 'generation' in stats
        assert 'population_size' in stats
        assert 'diversity' in stats

    def test_save_population(self, engine, fitness_function):
        """Test saving population."""
        engine.initialize_population()
        engine.evolve(fitness_function, generations=2)

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            filepath = f.name

        try:
            engine.save_population(filepath)

            assert os.path.exists(filepath)

            # Verify file contents
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)

            assert 'generation' in data
            assert 'population' in data

        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_load_population(self, engine, fitness_function):
        """Test loading population."""
        engine.initialize_population()
        engine.evolve(fitness_function, generations=2)

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            filepath = f.name

        try:
            engine.save_population(filepath)

            # Create new engine and load
            engine2 = EvolutionEngine(population_size=20)
            engine2.load_population(filepath)

            assert engine2.generation == engine.generation
            assert len(engine2.population) == len(engine.population)

        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_load_nonexistent_file(self, engine):
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            engine.load_population("nonexistent.json")

    def test_file_permission_error(self, engine):
        """Test file permission errors."""
        # Try to save to read-only location
        if os.name != 'nt':  # Skip on Windows
            with pytest.raises(PermissionError):
                engine.save_population("/root/test.json")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
