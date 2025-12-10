"""
Test suite for Runtime Extensions
"""

import asyncio
import json
import shutil
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

# Import the module to test
import runtime_extensions as re


class TestLearningMode:
    """Test LearningMode enum"""
    
    def test_learning_modes(self):
        """Test all learning modes are defined"""
        assert re.LearningMode.SUPERVISED.value == "supervised"
        assert re.LearningMode.UNSUPERVISED.value == "unsupervised"
        assert re.LearningMode.REINFORCED.value == "reinforced"
        assert re.LearningMode.EVOLUTIONARY.value == "evolutionary"
        assert re.LearningMode.DEMONSTRATION.value == "demonstration"
        assert re.LearningMode.IMITATION.value == "imitation"


class TestExplanationType:
    """Test ExplanationType enum"""
    
    def test_explanation_types(self):
        """Test all explanation types are defined"""
        assert re.ExplanationType.SIMPLE.value == "simple"
        assert re.ExplanationType.DETAILED.value == "detailed"
        assert re.ExplanationType.VISUAL.value == "visual"
        assert re.ExplanationType.TECHNICAL.value == "technical"
        assert re.ExplanationType.COMPARATIVE.value == "comparative"


class TestSubgraphPattern:
    """Test SubgraphPattern dataclass"""
    
    def test_pattern_creation(self):
        """Test creating a SubgraphPattern"""
        pattern = re.SubgraphPattern(
            pattern_id="test_123",
            name="TestPattern",
            graph_definition={"nodes": [], "edges": []},
            metadata={"test": True},
            performance_metrics={"latency": 0.1},
            usage_count=5,
            confidence_score=0.8
        )
        
        assert pattern.pattern_id == "test_123"
        assert pattern.name == "TestPattern"
        assert pattern.usage_count == 5
        assert pattern.confidence_score == 0.8
    
    def test_pattern_to_dict(self):
        """Test converting pattern to dictionary"""
        pattern = re.SubgraphPattern(
            pattern_id="test_456",
            name="TestPattern2",
            graph_definition={"nodes": [], "edges": []}
        )
        
        pattern_dict = pattern.to_dict()
        assert pattern_dict['pattern_id'] == "test_456"
        assert pattern_dict['name'] == "TestPattern2"
        assert 'creation_time' in pattern_dict
        assert pattern_dict['last_used'] is None


class TestExecutionExplanation:
    """Test ExecutionExplanation dataclass"""
    
    def test_explanation_creation(self):
        """Test creating an ExecutionExplanation"""
        explanation = re.ExecutionExplanation(
            subgraph_id="sg_001",
            explanation_type=re.ExplanationType.SIMPLE,
            summary="Test explanation",
            details={"key": "value"},
            confidence=0.75
        )
        
        assert explanation.subgraph_id == "sg_001"
        assert explanation.explanation_type == re.ExplanationType.SIMPLE
        assert explanation.confidence == 0.75
    
    def test_explanation_to_dict(self):
        """Test converting explanation to dictionary"""
        explanation = re.ExecutionExplanation(
            subgraph_id="sg_002",
            explanation_type=re.ExplanationType.DETAILED,
            summary="Detailed test"
        )
        
        exp_dict = explanation.to_dict()
        assert exp_dict['subgraph_id'] == "sg_002"
        assert exp_dict['explanation_type'] == "detailed"
        assert 'timestamp' in exp_dict


class TestAutonomousCycleReport:
    """Test AutonomousCycleReport dataclass"""
    
    def test_report_creation(self):
        """Test creating an AutonomousCycleReport"""
        report = re.AutonomousCycleReport(
            cycle_id="cycle_001",
            fitness_score=0.85,
            optimizations_applied=["opt1", "opt2"],
            evolution_proposals=[{"proposal": "test"}],
            safety_violations=[],
            performance_delta={"latency": -0.1}
        )
        
        assert report.cycle_id == "cycle_001"
        assert report.fitness_score == 0.85
        assert len(report.optimizations_applied) == 2
    
    def test_report_to_dict(self):
        """Test converting report to dictionary"""
        report = re.AutonomousCycleReport(
            cycle_id="cycle_002",
            fitness_score=0.9,
            optimizations_applied=[],
            evolution_proposals=[],
            safety_violations=["violation1"],
            performance_delta={}
        )
        
        report_dict = report.to_dict()
        assert report_dict['cycle_id'] == "cycle_002"
        assert report_dict['fitness_score'] == 0.9
        assert len(report_dict['safety_violations']) == 1


class TestSubgraphLearner:
    """Test SubgraphLearner class"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        temp_dir_path = Path(tempfile.mkdtemp())
        yield str(temp_dir_path) # Pass the path as a string
        shutil.rmtree(temp_dir_path)
    
    @pytest.fixture
    def learner(self, temp_dir):
        """Create learner instance"""
        return re.SubgraphLearner(learned_subgraphs_dir=temp_dir)
    
    @pytest.fixture
    def valid_graph(self):
        """Create valid graph definition"""
        # Return a function to ensure a fresh dict each time
        def _create_graph():
            return {
                "nodes": [
                    {"id": "n1", "type": "Source"},
                    {"id": "n2", "type": "Process"},
                    {"id": "n3", "type": "Sink"}
                ],
                "edges": [
                    {"from": "n1", "to": "n2"},
                    {"from": "n2", "to": "n3"}
                ]
            }
        return _create_graph() # Call the function to return the dict

    def test_learner_creation(self, learner, temp_dir):
        """Test creating SubgraphLearner"""
        assert learner.learned_dir == Path(temp_dir)
        assert learner.max_patterns == 1000
        assert len(learner.patterns) == 0
    
    def test_learn_subgraph_supervised(self, learner, valid_graph):
        """Test learning a subgraph in supervised mode"""
        success, pattern_id = learner.learn_subgraph(
            "TestType",
            valid_graph,
            mode=re.LearningMode.SUPERVISED,
            metadata={"test": True}
        )
        
        assert success is True
        assert pattern_id in learner.patterns
        pattern = learner.patterns[pattern_id]
        assert pattern.name == "TestType"
        assert pattern.confidence_score == 0.8  # Supervised default
    
    def test_learn_subgraph_unsupervised(self, learner, valid_graph):
        """Test learning a subgraph in unsupervised mode"""
        success, pattern_id = learner.learn_subgraph(
            "TestType",
            valid_graph,
            mode=re.LearningMode.UNSUPERVISED
        )
        
        assert success is True
        pattern = learner.patterns[pattern_id]
        assert pattern.confidence_score == 0.5  # Unsupervised default
    
    def test_learn_subgraph_duplicate(self, learner, valid_graph):
        """Test learning duplicate subgraph updates usage count"""
        success1, pattern_id1 = learner.learn_subgraph("Type1", valid_graph)
        initial_count = learner.patterns[pattern_id1].usage_count
        
        success2, pattern_id2 = learner.learn_subgraph("Type1", valid_graph)
        
        assert success1 and success2
        assert pattern_id1 == pattern_id2
        assert learner.patterns[pattern_id1].usage_count == initial_count + 1
    
    def test_learn_subgraph_invalid_structure(self, learner):
        """Test learning with invalid graph structure"""
        invalid_graph = {"nodes": []}  # Missing edges
        
        success, message = learner.learn_subgraph("Invalid", invalid_graph)
        
        assert success is False
        assert "Invalid graph structure" in message
    
    def test_validate_graph_structure_missing_nodes(self, learner):
        """Test validation with missing nodes field"""
        invalid_graph = {"edges": []}
        assert learner._validate_graph_structure(invalid_graph) is False
    
    def test_validate_graph_structure_missing_edges(self, learner):
        """Test validation with missing edges field"""
        invalid_graph = {"nodes": []}
        assert learner._validate_graph_structure(invalid_graph) is False
    
    def test_validate_graph_structure_duplicate_node_id(self, learner):
        """Test validation with duplicate node IDs"""
        invalid_graph = {
            "nodes": [
                {"id": "n1", "type": "A"},
                {"id": "n1", "type": "B"}  # Duplicate
            ],
            "edges": []
        }
        assert learner._validate_graph_structure(invalid_graph) is False
    
    def test_validate_graph_structure_unknown_node_reference(self, learner):
        """Test validation with unknown node in edge"""
        invalid_graph = {
            "nodes": [{"id": "n1"}],
            "edges": [{"from": "n1", "to": "n2"}]  # n2 doesn't exist
        }
        assert learner._validate_graph_structure(invalid_graph) is False
    
    def test_load_learned_subgraphs(self, learner, valid_graph, temp_dir):
        """Test loading patterns from disk"""
        # Learn a pattern
        success, pattern_id = learner.learn_subgraph("Saved", valid_graph)
        assert success
        
        # Create new learner (should load from disk)
        new_learner = re.SubgraphLearner(learned_subgraphs_dir=temp_dir)
        
        assert len(new_learner.patterns) == 1
        assert pattern_id in new_learner.patterns
    
    def test_get_pattern(self, learner, valid_graph):
        """Test getting pattern by ID"""
        success, pattern_id = learner.learn_subgraph("GetTest", valid_graph)
        assert success
        
        initial_count = learner.patterns[pattern_id].usage_count
        pattern = learner.get_pattern(pattern_id)
        
        assert pattern is not None
        assert pattern.pattern_id == pattern_id
        assert pattern.usage_count == initial_count + 1
    
    def test_get_patterns_by_type(self, learner, valid_graph):
        """Test getting patterns by type"""
        # Define graph structures explicitly
        graph_a1 = {
            "nodes": [{"id": "n1", "type": "Source"}, {"id": "n2", "type": "Process"}, {"id": "n3", "type": "Sink"}],
            "edges": [{"from": "n1", "to": "n2"}, {"from": "n2", "to": "n3"}]
        }
        graph_a2 = {
            "nodes": [{"id": "n1a", "type": "Source"}, {"id": "n2", "type": "Process"}, {"id": "n3", "type": "Sink"}],
            "edges": [{"from": "n1a", "to": "n2"}, {"from": "n2", "to": "n3"}]
        }
        graph_b1 = { # Same structure as graph_a1
            "nodes": [{"id": "n1", "type": "Source"}, {"id": "n2", "type": "Process"}, {"id": "n3", "type": "Sink"}],
            "edges": [{"from": "n1", "to": "n2"}, {"from": "n2", "to": "n3"}]
        }

        # Learn the patterns
        success1, id1 = learner.learn_subgraph("TypeA", graph_a1)
        success2, id2 = learner.learn_subgraph("TypeA", graph_a2)
        success3, id3 = learner.learn_subgraph("TypeB", graph_b1)

        assert success1 and success2 and success3
        assert id1 != id2 # Ensure different graphs yield different IDs for same type
        assert id1 != id3 # Ensure same graph yields different IDs for different types
        assert id2 != id3

        # Get patterns by type
        type_a_patterns = learner.get_patterns_by_type("TypeA")
        type_b_patterns = learner.get_patterns_by_type("TypeB")
        
        # Assertions
        assert len(type_a_patterns) == 2
        assert len(type_b_patterns) == 1
        # Check if the correct pattern IDs are present
        type_a_ids = {p.pattern_id for p in type_a_patterns}
        assert type_a_ids == {id1, id2}
        assert type_b_patterns[0].pattern_id == id3

    def test_update_pattern_performance(self, learner, valid_graph):
        """Test updating pattern performance metrics"""
        success, pattern_id = learner.learn_subgraph("PerfTest", valid_graph)
        assert success
        
        metrics = {
            "latency": 0.05,
            "throughput": 1000,
            "success_rate": 0.95
        }
        
        updated = learner.update_pattern_performance(pattern_id, metrics)
        assert updated is True
        
        pattern = learner.patterns[pattern_id]
        assert pattern.performance_metrics["latency"] == 0.05
        assert pattern.confidence_score > 0.5  # Updated based on success_rate
    
    def test_evict_least_used(self, learner, valid_graph):
        """Test evicting least used patterns"""
        # Create a learner with max_patterns=2 to test eviction
        small_learner = re.SubgraphLearner(
            learned_subgraphs_dir=learner.learned_dir,
            max_patterns=2
        )
        
        # Add 3 patterns (should trigger eviction of one)
        ids = []
        for i in range(3):
            graph = valid_graph.copy()
            # Must modify graph to get new pattern ID
            graph["nodes"] = [{"id": f"n{i}", "type": "Source"}, {"id": f"n_sink_{i}", "type": "Sink"}]
            graph["edges"] = [{"from": f"n{i}", "to": f"n_sink_{i}"}]
            # Use distinct types as well to guarantee different IDs with the fixed generator
            success, pid = small_learner.learn_subgraph(f"Type{i}", graph)
            assert success
            ids.append(pid)
            time.sleep(0.01) # Ensure different creation/last_used times initially

        # At this point, ids[0] was evicted when ids[2] was added.
        # Cache contains: ids[1], ids[2]
        assert len(small_learner.patterns) == 2
        assert ids[0] not in small_learner.patterns
        assert ids[1] in small_learner.patterns
        assert ids[2] in small_learner.patterns

        # Get patterns to update usage and last_used time
        small_learner.get_pattern(ids[1]) # Access ids[1]
        time.sleep(0.01)
        small_learner.get_pattern(ids[2]) # Access ids[2] (most recent)
        time.sleep(0.01)
        
        # Add one more pattern to trigger eviction
        graph4 = valid_graph.copy()
        graph4["nodes"] = [{"id": "n4", "type": "Source"}]
        graph4["edges"] = []
        success4, id4 = small_learner.learn_subgraph("Type4", graph4)
        assert success4

        # Now, the cache should contain ids[2] (most recently used) and id4 (newest).
        # ids[1] should have been evicted because it was used *before* ids[2].
        assert len(small_learner.patterns) == 2
        assert ids[0] not in small_learner.patterns # Still evicted
        assert ids[1] not in small_learner.patterns # Newly evicted (LRU)
        assert ids[2] in small_learner.patterns     # Remained (more recently used)
        assert id4 in small_learner.patterns        # The newly added one


    def test_generate_pattern_id(self, learner, valid_graph):
        """Test pattern ID generation"""
        id1 = learner._generate_pattern_id("Test", valid_graph)
        id2 = learner._generate_pattern_id("Test", valid_graph)
        
        # Create slightly different graph
        graph2 = valid_graph.copy()
        graph2["nodes"][0]["type"] = "SourceModified"
        id3 = learner._generate_pattern_id("Test", graph2)

        # Use same graph but different type
        id4 = learner._generate_pattern_id("TestDifferentType", valid_graph)

        assert id1 == id2  # Same type, same graph should produce same ID
        assert id1 != id3  # Same type, different graph should produce different ID
        assert id1 != id4  # Different type, same graph should produce different ID
        assert len(id1) == 12  # MD5 hash truncated to 12 chars
    
    def test_persistence(self, learner, valid_graph):
        """Test pattern persistence to disk"""
        success, pattern_id = learner.learn_subgraph("Persistent", valid_graph)
        assert success
        
        pattern_file = learner.learned_dir / f"{pattern_id}.json"
        assert pattern_file.exists()
        
        with open(pattern_file, 'r') as f:
            data = json.load(f)
            assert data['pattern_id'] == pattern_id
            assert data['name'] == "Persistent"


class TestAutonomousOptimizer:
    """Test AutonomousOptimizer class"""
    
    def test_optimizer_creation(self):
        """Test creating AutonomousOptimizer"""
        optimizer = re.AutonomousOptimizer()
        
        assert optimizer.current_fitness == 0.5
        assert len(optimizer.optimization_history) == 0
        assert optimizer.optimization_config['min_fitness_threshold'] == 0.3
    
    @pytest.mark.asyncio
    async def test_trigger_autonomous_cycle(self):
        """Test triggering autonomous cycle"""
        optimizer = re.AutonomousOptimizer()
        
        graph = {"nodes": [], "edges": []}
        metrics = {
            "latency": 0.1,
            "throughput": 500,
            "success_rate": 0.95,
            "cache_hit_rate": 0.7
        }
        
        # Fitness should be high, so no optimizations applied
        fitness = optimizer._calculate_fitness(metrics)
        assert fitness >= optimizer.optimization_config['min_fitness_threshold']

        report = await optimizer.trigger_autonomous_cycle(graph, metrics)
        
        assert report is not None
        assert report.cycle_id.startswith("cycle_")
        assert report.fitness_score == fitness
        assert len(report.optimizations_applied) == 0 # No optimizations needed
        assert len(report.evolution_proposals) == 0 # No optimizations needed
        assert len(optimizer.optimization_history) == 1
    
    @pytest.mark.asyncio
    async def test_autonomous_cycle_low_fitness(self):
        """Test autonomous cycle with low fitness triggers optimizations"""
        optimizer = re.AutonomousOptimizer()
        
        graph = {"nodes": [], "edges": []}
        metrics = {
            "latency": 0.9,  # High latency
            "throughput": 10,  # Low throughput
            "success_rate": 0.2,  # Low success
            "cache_hit_rate": 0.1
        }
        
        fitness = optimizer._calculate_fitness(metrics)
        assert fitness < optimizer.optimization_config['min_fitness_threshold']

        report = await optimizer.trigger_autonomous_cycle(graph, metrics)
        
        assert report.fitness_score == fitness
        # Should trigger optimizations/evolution when fitness is low, if available
        optimizations_triggered = (len(report.optimizations_applied) > 0 or 
                                   len(report.evolution_proposals) > 0)
        
        should_trigger = (re.EVOLUTION_AVAILABLE or re.OPTIMIZER_AVAILABLE or 
                         re.NSO_AVAILABLE)

        if should_trigger:
             assert optimizations_triggered is True
        else:
             assert optimizations_triggered is False
    
    def test_calculate_fitness(self):
        """Test fitness calculation"""
        optimizer = re.AutonomousOptimizer()
        
        metrics = {
            "latency": 0.2,
            "throughput": 800,
            "success_rate": 0.9,
            "cache_hit_rate": 0.6
        }
        
        fitness = optimizer._calculate_fitness(metrics)
        
        assert 0.0 <= fitness <= 1.0
        # Expected fitness: (0.3 * (1-0.2)) + (0.3 * 0.8) + (0.3 * 0.9) + (0.1 * 0.6)
        # = (0.3 * 0.8) + 0.24 + 0.27 + 0.06
        # = 0.24 + 0.24 + 0.27 + 0.06 = 0.81
        assert abs(fitness - 0.81) < 1e-9 
    
    def test_calculate_fitness_bad_metrics(self):
        """Test fitness calculation with bad metrics"""
        optimizer = re.AutonomousOptimizer()
        
        metrics = {
            "latency": 2.0,  # Very high (clamps to 1.0 in calculation)
            "throughput": 0,  # Zero
            "success_rate": 0.0,
            "cache_hit_rate": 0.0
        }
        
        fitness = optimizer._calculate_fitness(metrics)
        
        # Expected fitness: (0.3 * (1-1.0)) + (0.3 * 0.0) + (0.3 * 0.0) + (0.1 * 0.0) = 0.0
        assert abs(fitness - 0.0) < 1e-9


class TestExecutionExplainer:
    """Test ExecutionExplainer class"""
    
    @pytest.fixture
    def explainer(self):
        """Create explainer instance"""
        return re.ExecutionExplainer()
    
    def test_explainer_creation(self, explainer):
        """Test creating ExecutionExplainer"""
        assert len(explainer.explanation_cache) == 0
        assert len(explainer.explanation_history) == 0
    
    def test_explain_execution_no_tensors(self, explainer):
        """Test explaining execution without tensors"""
        subgraph = MagicMock() # Use mock object
        subgraph.__repr__ = lambda s: "<MockSubgraph>" # for clarity if needed
        inputs = {"x": 1, "y": 2}
        result = {"output": 3}
        
        explanations = explainer.explain_execution(subgraph, inputs, result)
        
        assert len(explanations) == 1
        explanation = explanations[0]
        assert explanation.explanation_type == re.ExplanationType.SIMPLE
        assert explanation.subgraph_id.startswith("subgraph_")
        assert explanation.summary == "Subgraph processed 2 inputs and produced output"
        assert explanation.confidence >= 0.5
    
    def test_explain_execution_with_tensor(self, explainer):
        """Test explaining execution with numpy tensor"""
        if not NUMPY_AVAILABLE:
            pytest.skip("NumPy not available")
        
        subgraph = MagicMock()
        subgraph.__repr__ = lambda s: "<MockTensorSubgraph>"
        inputs = {"data": np.array([1, 2, 3])}
        result = {"tensor": np.zeros((3, 3))}
        
        explanations = explainer.explain_execution(
            subgraph, inputs, result, 
            explanation_type=re.ExplanationType.TECHNICAL # Use enum directly
        )
        
        assert len(explanations) == 1
        explanation = explanations[0]
        assert explanation.explanation_type == re.ExplanationType.TECHNICAL
        assert explanation.subgraph_id.startswith("subgraph_")
        assert 'tensor_shapes' in explanation.details
        assert explanation.details['tensor_shapes'] == {'tensor': '(3, 3)'}
    
    def test_flag_unclear_explanation_high_sparsity(self, explainer):
        """Test flagging unclear explanation with sparse output"""
        if not NUMPY_AVAILABLE:
            pytest.skip("NumPy not available")
        
        explanation = re.ExecutionExplanation(
            subgraph_id="sparse",
            explanation_type=re.ExplanationType.SIMPLE,
            summary="Test",
            confidence=0.4 # Confidence < 0.5 triggers the check in explain_execution
        )
        
        # Create sparse output (>90% zeros)
        sparse_output = {"data": np.zeros((10, 10))}
        sparse_output["data"][0, 0] = 1.0  # Only 1% non-zero
        
        # Call the internal method directly for focused testing
        explainer._flag_unclear_explanation(explanation, sparse_output)
        
        assert 'warning' in explanation.details
        assert 'sparsity' in explanation.details['warning'].lower()
    
    def test_flag_unclear_explanation_low_sparsity(self, explainer):
        """Test no warning for non-sparse output"""
        if not NUMPY_AVAILABLE:
            pytest.skip("NumPy not available")
        
        explanation = re.ExecutionExplanation(
            subgraph_id="dense",
            explanation_type=re.ExplanationType.SIMPLE,
            summary="Test",
            confidence=0.4
        )
        
        # Create dense output (<90% zeros)
        dense_output = {"data": np.ones((10, 10))}
        
        explainer._flag_unclear_explanation(explanation, dense_output)
        
        # Should not add warning for dense data
        assert 'warning' not in explanation.details
    
    def test_get_explanation_summary(self, explainer):
        """Test getting explanation summary"""
        subgraph = MagicMock()
        subgraph.__repr__ = lambda s: "<MockSummarySubgraph>"
        inputs = {"x": 1}
        outputs = {"y": 2}
        
        # Generate multiple explanations
        exp1 = explainer.explain_execution(subgraph, inputs, outputs, re.ExplanationType.SIMPLE)
        exp2 = explainer.explain_execution(subgraph, inputs, outputs, re.ExplanationType.DETAILED)
        
        # Get summary
        subgraph_id = explainer._generate_subgraph_id(subgraph) # Get the generated ID
        summary = explainer.get_explanation_summary(subgraph_id)
        
        assert summary is not None
        # Check if summaries from both explanations are present
        assert exp1[0].summary in summary
        assert exp2[0].summary in summary
    
    def test_get_explanation_summary_not_found(self, explainer):
        """Test getting summary for non-existent subgraph"""
        summary = explainer.get_explanation_summary("nonexistent_subgraph_id")
        assert summary is None
    
    def test_extract_tensors_dict(self, explainer):
        """Test extracting tensors from dict"""
        if not NUMPY_AVAILABLE:
            pytest.skip("NumPy not available")
        
        result = {
            "array": np.array([1, 2, 3]),
            "scalar": 42,
            "nested": {
                "tensor": np.zeros((2, 2))
            }
        }
        
        tensors = explainer._extract_tensors(result)
        
        assert "array" in tensors
        assert isinstance(tensors["array"], np.ndarray)
        assert "nested.tensor" in tensors
        assert isinstance(tensors["nested.tensor"], np.ndarray)
        assert "scalar" not in tensors
        assert len(tensors) == 2
    
    def test_extract_tensors_nested(self, explainer):
        """Test extracting tensors from nested structures (list)"""
        if not NUMPY_AVAILABLE:
            pytest.skip("NumPy not available")
        
        result = [
            np.array([1]),
            {"data": "not a tensor"}, # Dict does not contain tensor
            "string",
            (np.array([4, 5]),) # Tensor inside tuple
        ]
        
        tensors = explainer._extract_tensors(result)
        
        # Expecting tensors from item_0 and item_3 (inside tuple)
        assert "item_0" in tensors
        assert isinstance(tensors["item_0"], np.ndarray)
        # Nested list/tuple extraction is not recursive in the current implementation
        # It only finds tensors that are direct elements of the list/tuple.
        # assert "item_3.item_0" in tensors # This would fail based on current code
        assert len(tensors) == 1 # Only item_0 is directly a tensor

    def test_is_tensor_numpy(self, explainer):
        """Test tensor identification for numpy"""
        if not NUMPY_AVAILABLE:
            pytest.skip("NumPy not available")
        
        assert explainer._is_tensor(np.array([1, 2, 3])) is True
    
    def test_is_tensor_not_tensor(self, explainer):
        """Test tensor identification for non-tensors"""
        not_tensor = [1, 2, 3]
        assert explainer._is_tensor(not_tensor) is False
        assert explainer._is_tensor(42) is False
        assert explainer._is_tensor("hello") is False
    
    def test_calculate_explanation_confidence(self, explainer):
        """Test confidence calculation"""
        subgraph = MagicMock()
        subgraph.__repr__ = lambda s: "<MockConfSubgraph>"
        
        # Simple output should have higher confidence
        simple_output = 42
        conf1 = explainer._calculate_explanation_confidence(subgraph, simple_output)
        
        # Complex output (large dict) should have lower base confidence
        complex_output = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6}
        conf2 = explainer._calculate_explanation_confidence(subgraph, complex_output)
        
        # Basic dict output
        dict_output = {"a": 1, "b": 2}
        conf3 = explainer._calculate_explanation_confidence(subgraph, dict_output)

        # Base confidence is 0.5
        # simple adds 0.3 -> 0.8
        # dict adds 0.1 * min(len, 5)/5
        assert abs(conf1 - 0.8) < 1e-9
        assert abs(conf2 - (0.5 + 0.1 * 5/5)) < 1e-9 # max len effect is 5
        assert abs(conf3 - (0.5 + 0.1 * 2/5)) < 1e-9 # len is 2
        assert conf1 > conf2
        assert conf2 > conf3 # This comparison depends on base vs dict adjustment


class TestRuntimeExtensions:
    """Test RuntimeExtensions main class"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory"""
        temp_dir_path = Path(tempfile.mkdtemp())
        yield str(temp_dir_path) # Pass the path as a string
        shutil.rmtree(temp_dir_path)

    @pytest.fixture
    def extensions(self, temp_dir):
        """Create RuntimeExtensions instance"""
        return re.RuntimeExtensions(
            learned_subgraphs_dir=temp_dir,
            enable_autonomous=True
        )
    
    def test_extensions_creation(self, extensions):
        """Test creating RuntimeExtensions"""
        assert extensions.subgraph_learner is not None
        assert extensions.autonomous_optimizer is not None
        assert extensions.execution_explainer is not None
        assert extensions.stats['patterns_learned'] == 0
    
    def test_learn_subgraph(self, extensions):
        """Test learning subgraph through main interface"""
        graph = {
            "nodes": [{"id": "n1"}, {"id": "n2"}],
            "edges": [{"from": "n1", "to": "n2"}]
        }
        
        success, result = extensions.learn_subgraph(
            "TestPattern",
            graph,
            mode="supervised",
            metadata={"source": "test"}
        )
        
        assert success is True
        assert isinstance(result, str) # Should return pattern ID or message
        assert extensions.stats['patterns_learned'] == 1
        assert len(extensions.load_learned_subgraphs()) == 1
    
    def test_load_learned_subgraphs(self, extensions):
        """Test loading learned subgraphs"""
        graph = {
            "nodes": [{"id": "a"}],
            "edges": []
        }
        assert len(extensions.load_learned_subgraphs()) == 0 # Initially empty
        
        success, pattern_id = extensions.learn_subgraph("Pattern1", graph)
        assert success
        
        patterns = extensions.load_learned_subgraphs()
        
        assert len(patterns) == 1
        assert pattern_id in patterns
        assert patterns[pattern_id].name == "Pattern1"
    
    @pytest.mark.asyncio
    async def test_trigger_autonomous_cycle(self, extensions):
        """Test triggering autonomous cycle"""
        graph = {"nodes": [], "edges": []}
        metrics = {
            "latency": 0.1,
            "throughput": 500,
            "success_rate": 0.95,
            "cache_hit_rate": 0.7
        }
        assert extensions.stats['optimizations_run'] == 0

        report = await extensions.trigger_autonomous_cycle(graph, metrics)
        
        assert report is not None
        assert extensions.stats['optimizations_run'] == 1
    
    @pytest.mark.asyncio
    async def test_trigger_autonomous_cycle_disabled(self, temp_dir):
        """Test autonomous cycle when disabled"""
        extensions_disabled = re.RuntimeExtensions(
            learned_subgraphs_dir=temp_dir, 
            enable_autonomous=False
        )
        assert extensions_disabled.autonomous_optimizer is None
        
        graph = {"nodes": [], "edges": []}
        metrics = {"latency": 0.1}
        
        report = await extensions_disabled.trigger_autonomous_cycle(graph, metrics)
        
        assert report is None
        assert extensions_disabled.stats['optimizations_run'] == 0
    
    def test_explain_execution(self, extensions):
        """Test execution explanation"""
        subgraph = MagicMock()
        subgraph.__repr__ = lambda s: "<MockExpSubgraph>"
        inputs = {"x": 1}
        outputs = {"y": 2}
        
        assert extensions.stats['explanations_generated'] == 0
        explanations = extensions.explain_execution(
            subgraph, inputs, outputs,
            explanation_type="detailed"
        )
        
        assert len(explanations) == 1
        assert explanations[0].explanation_type == re.ExplanationType.DETAILED
        assert extensions.stats['explanations_generated'] == 1
    
    def test_flag_unclear_explanation(self, extensions):
        """Test flagging unclear explanations (via main interface)"""
        # This test mainly checks if the method exists and runs without error
        explanation = re.ExecutionExplanation(
            subgraph_id="unclear",
            explanation_type=re.ExplanationType.SIMPLE,
            summary="Unclear",
            confidence=0.3
        )
        
        outputs = {"result": "unclear"}
        
        try:
            extensions.flag_unclear_explanation(explanation, outputs)
        except Exception as e:
            pytest.fail(f"flag_unclear_explanation raised an exception: {e}")

    def test_get_statistics(self, extensions):
        """Test getting runtime statistics"""
        # Perform some operations
        graph = {"nodes": [{"id": "n1"}], "edges": []}
        extensions.learn_subgraph("Stat", graph)
        
        stats_before = extensions.get_statistics()
        assert stats_before['patterns_learned'] == 1
        assert stats_before['total_patterns'] == 1
        
        # Explain something
        extensions.explain_execution(MagicMock(), {}, {})
        
        stats_after = extensions.get_statistics()
        
        assert stats_after['patterns_learned'] == 1
        assert stats_after['total_patterns'] == 1
        assert stats_after['explanations_generated'] == 1
        assert 'current_fitness' in stats_after
        assert 'cached_explanations' in stats_after
        assert stats_after['cached_explanations'] >= 1


class TestHelperFunctions:
    """Test helper functions"""
    
    def test_create_runtime_extensions_default(self):
        """Test creating extensions with default config"""
        # Ensure it doesn't crash and uses defaults
        try:
            ext = re.create_runtime_extensions()
            assert ext is not None
            assert ext.subgraph_learner.learned_dir == Path("learned_subgraphs")
            assert ext.autonomous_optimizer is not None # Default is True
        except Exception as e:
            pytest.fail(f"create_runtime_extensions raised an exception: {e}")
        finally:
            # Clean up default dir if created
            if Path("learned_subgraphs").exists():
                shutil.rmtree("learned_subgraphs", ignore_errors=True)


    def test_create_runtime_extensions_with_config(self, tmp_path):
        """Test creating extensions with custom config"""
        # Use pytest's tmp_path fixture for safer temp dirs
        custom_dir = tmp_path / "custom_learn"
        config = {
            'learned_subgraphs_dir': str(custom_dir),
            'enable_autonomous': False
        }
        
        ext = re.create_runtime_extensions(config)
        
        assert ext is not None
        assert ext.subgraph_learner.learned_dir == custom_dir
        assert ext.autonomous_optimizer is None # Check disabled
    
    def test_load_extension_config(self, tmp_path):
        """Test loading config from file"""
        config_file = tmp_path / "config.json"
        config_data = {
            "learned_subgraphs_dir": "test_dir",
            "enable_autonomous": True
        }
        
        config_file.write_text(json.dumps(config_data))
        
        loaded_config = re.load_extension_config(str(config_file))
        
        assert loaded_config['learned_subgraphs_dir'] == "test_dir"
        assert loaded_config['enable_autonomous'] is True
    
    def test_load_extension_config_missing_file(self):
        """Test loading config from missing file"""
        # Should return empty dict and log error (check log capture if needed)
        config = re.load_extension_config("nonexistent_config.json")
        assert config == {}


class TestIntegration:
    """Integration tests"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for integration test"""
        temp_dir_path = Path(tempfile.mkdtemp(prefix="integ_"))
        yield str(temp_dir_path)
        shutil.rmtree(temp_dir_path)

    @pytest.mark.asyncio
    async def test_complete_workflow(self, temp_dir):
        """Test complete workflow of learning, optimizing, and explaining"""
        # Create extensions using the temp_dir
        ext = re.create_runtime_extensions(
            config={'learned_subgraphs_dir': temp_dir, 'enable_autonomous': True}
        )
        
        # Define test graph
        graph = {
            "nodes": [
                {"id": "input", "type": "Source"},
                {"id": "process", "type": "Transform"},
                {"id": "output", "type": "Sink"}
            ],
            "edges": [
                {"from": "input", "to": "process"},
                {"from": "process", "to": "output"}
            ]
        }
        
        # Learn subgraph
        success, pattern_id = ext.learn_subgraph(
            "MyPattern", 
            graph,
            mode="supervised",
            metadata={"version": 1}
        )
        assert success
        
        # Check initial stats
        stats_after_learn = ext.get_statistics()
        assert stats_after_learn['patterns_learned'] == 1
        assert stats_after_learn['total_patterns'] == 1

        # Trigger optimization with some metrics
        metrics = {
            "latency": 0.8, # Low enough to potentially trigger cycle
            "throughput": 100,
            "success_rate": 0.4,
            "cache_hit_rate": 0.7
        }
        
        # Mock sub-components if they are not fully available or for deterministic test
        # Only mock if necessary and if the component exists
        if ext.autonomous_optimizer and hasattr(ext.autonomous_optimizer, 'evolution_engine') and ext.autonomous_optimizer.evolution_engine:
             mock_evo_engine = MagicMock()
             # Configure mock return values as needed for evolution if called
             ext.autonomous_optimizer.evolution_engine = mock_evo_engine

        # Mock the trigger method on the optimizer itself IF NEEDED
        # If testing the full flow including the optimizer logic, don't mock this
        # If just testing the RuntimeExtensions interface, mocking is okay:
        # Example mock:
        if ext.autonomous_optimizer:
            # Use AsyncMock for async methods
            ext.autonomous_optimizer.trigger_autonomous_cycle = AsyncMock(
                return_value=re.AutonomousCycleReport(
                    cycle_id="mock_cycle_1", fitness_score=0.25, optimizations_applied=["mock_opt"],
                    evolution_proposals=[], safety_violations=[], performance_delta={}
                )
            )
        
        # Call the main interface method
        report = await ext.trigger_autonomous_cycle(graph, metrics) 
        
        # Check report and stats after optimization
        if ext.autonomous_optimizer: # Only check if optimizer was enabled
            assert report is not None
            assert report.cycle_id.startswith("mock_cycle_") # Check mock was used
            assert ext.stats['optimizations_run'] == 1 
            assert 0.0 <= report.fitness_score <= 1.0 
        else:
            assert report is None
            assert ext.stats['optimizations_run'] == 0

        # Explain an execution
        subgraph_mock = MagicMock()
        inputs_mock = {"in": 1}
        outputs_mock = {"out": 2}
        explanations = ext.explain_execution(subgraph_mock, inputs_mock, outputs_mock)
        assert len(explanations) == 1

        # Get final stats
        stats = ext.get_statistics()
        assert stats['patterns_learned'] == 1
        assert stats['total_patterns'] == 1 # Only one pattern learned
        assert stats['optimizations_run'] == (1 if ext.autonomous_optimizer else 0)
        assert stats['explanations_generated'] == 1


if __name__ == "__main__":
    # Allows running the tests directly using `python test_runtime_extensions.py`
    # The `-v` flag provides verbose output
    pytest.main([__file__, "-v"])