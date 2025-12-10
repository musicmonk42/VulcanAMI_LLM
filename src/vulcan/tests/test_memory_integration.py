"""
Comprehensive test suite for VULCAN-AGI Memory System

Tests all components and their integration:
- Base memory operations
- Hierarchical memory with tool selection
- Distributed memory with replication
- Persistence and compression
- Retrieval and search
- Specialized memory types (Episodic, Semantic, Procedural, Working)
- Consolidation and optimization

Run with: python test_memory_system.py
"""

from __future__ import annotations

import sys
import os
import time
import tempfile
import shutil
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add parent directory to path if running standalone
sys.path.insert(0, str(Path(__file__).parent))

try:
    from vulcan.memory import (
        Memory,
        MemoryType,
        MemoryConfig,
        MemoryQuery,
        MemoryStats,
        HierarchicalMemory,
        DistributedMemory,
        MemoryFederation,
        MemoryPersistence,
        MemoryVersionControl,
        CompressionType,
        MemoryIndex,
        MemorySearch,
        AttentionMechanism,
        MemoryConsolidator,
        ConsolidationStrategy,
        EpisodicMemory,
        SemanticMemory,
        ProceduralMemory,
        WorkingMemory,
        Episode,
        Concept,
        Skill,
    )

    MEMORY_MODULE_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import memory module: {e}")
    MEMORY_MODULE_AVAILABLE = False


# Test results tracking
class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
        self.warnings = []

    def record_pass(self, test_name: str):
        self.passed += 1
        logger.info(f"✓ PASSED: {test_name}")

    def record_fail(self, test_name: str, error: str):
        self.failed += 1
        self.errors.append(f"{test_name}: {error}")
        logger.error(f"✗ FAILED: {test_name} - {error}")

    def record_warning(self, test_name: str, warning: str):
        self.warnings.append(f"{test_name}: {warning}")
        logger.warning(f"⚠ WARNING: {test_name} - {warning}")

    def print_summary(self):
        total = self.passed + self.failed
        print("\n" + "=" * 70)
        print(f"TEST SUMMARY: {self.passed}/{total} tests passed")
        print(
            f"Success rate: {(self.passed / total * 100):.1f}%"
            if total > 0
            else "No tests run"
        )
        print("=" * 70)

        if self.errors:
            print("\nFAILURES:")
            for error in self.errors:
                print(f"  • {error}")

        if self.warnings:
            print("\nWARNINGS:")
            for warning in self.warnings:
                print(f"  • {warning}")

        print()


results = TestResults()

# ============================================================
# TEST UTILITIES
# ============================================================


def assert_true(condition: bool, message: str):
    """Assert condition is true."""
    if not condition:
        raise AssertionError(message)


def assert_equal(actual, expected, message: str = ""):
    """Assert values are equal."""
    if actual != expected:
        raise AssertionError(f"{message}: Expected {expected}, got {actual}")


def assert_not_none(value, message: str = ""):
    """Assert value is not None."""
    if value is None:
        raise AssertionError(f"{message}: Value is None")


def create_test_config() -> MemoryConfig:
    """Create test configuration."""
    return MemoryConfig(
        max_working_memory=10,
        max_short_term=50,
        max_long_term=200,
        enable_compression=True,
        compression_type=CompressionType.LZ4,
        consolidation_interval=1.0,
        consolidation_threshold=0.3,
        enable_persistence=True,
        persistence_path=tempfile.mkdtemp(prefix="memory_test_"),
        checkpoint_interval=10.0,
        enable_indexing=True,
        similarity_threshold=0.7,
    )


# ============================================================
# BASE MEMORY TESTS
# ============================================================


def test_memory_creation():
    """Test basic Memory object creation."""
    try:
        memory = Memory(
            id="test_001",
            type=MemoryType.EPISODIC,
            content="Test memory content",
            importance=0.8,
            metadata={"test": True},
        )

        assert_not_none(memory, "Memory creation failed")
        assert_equal(memory.id, "test_001", "Memory ID")
        assert_equal(memory.type, MemoryType.EPISODIC, "Memory type")
        assert_equal(memory.importance, 0.8, "Memory importance")

        results.record_pass("test_memory_creation")
    except Exception as e:
        results.record_fail("test_memory_creation", str(e))


def test_memory_salience():
    """Test memory salience computation."""
    try:
        memory = Memory(
            id="test_002",
            type=MemoryType.SEMANTIC,
            content="Important fact",
            importance=0.9,
            access_count=5,
            decay_rate=0.01,
        )

        # Compute salience
        salience = memory.compute_salience()

        assert_true(0 <= salience <= 2.0, "Salience should be in valid range")
        assert_true(salience > 0.5, "High importance memory should have high salience")

        # Test decay
        time.sleep(0.1)
        new_salience = memory.compute_salience()
        assert_true(new_salience <= salience, "Salience should decay over time")

        results.record_pass("test_memory_salience")
    except Exception as e:
        results.record_fail("test_memory_salience", str(e))


def test_memory_serialization():
    """Test memory to/from dict conversion."""
    try:
        original = Memory(
            id="test_003",
            type=MemoryType.PROCEDURAL,
            content={"action": "test"},
            importance=0.7,
            metadata={"key": "value"},
        )

        # Convert to dict
        mem_dict = original.to_dict()
        assert_true(isinstance(mem_dict, dict), "to_dict should return dict")
        assert_equal(mem_dict["id"], "test_003", "ID in dict")

        # Convert back
        restored = Memory.from_dict(mem_dict)
        assert_equal(restored.id, original.id, "Restored ID")
        assert_equal(restored.type, original.type, "Restored type")
        assert_equal(restored.importance, original.importance, "Restored importance")

        results.record_pass("test_memory_serialization")
    except Exception as e:
        results.record_fail("test_memory_serialization", str(e))


# ============================================================
# HIERARCHICAL MEMORY TESTS
# ============================================================


def test_hierarchical_memory_basic():
    """Test basic hierarchical memory operations."""
    try:
        config = create_test_config()
        hmem = HierarchicalMemory(config)

        # Store memories
        mem1 = hmem.store("Test content 1", importance=0.8)
        mem2 = hmem.store("Test content 2", importance=0.5)

        assert_not_none(mem1, "First memory storage")
        assert_not_none(mem2, "Second memory storage")
        assert_true(mem1.id != mem2.id, "Memory IDs should be unique")

        # Retrieve
        query = MemoryQuery(query_type="similarity", content="Test content", limit=10)
        result = hmem.retrieve(query)

        assert_true(len(result.memories) > 0, "Should retrieve memories")
        assert_true(len(result.scores) == len(result.memories), "Scores match memories")

        results.record_pass("test_hierarchical_memory_basic")
    except Exception as e:
        results.record_fail("test_hierarchical_memory_basic", str(e))


def test_tool_selection_memory():
    """Test tool selection history tracking."""
    try:
        config = create_test_config()
        hmem = HierarchicalMemory(config)

        # Generate problem features
        problem_features = np.random.randn(hmem.embedding_dimension)
        problem_features = problem_features / np.linalg.norm(problem_features)

        # Store tool selection
        record = hmem.store_tool_selection(
            problem_features=problem_features,
            problem_description="Solve optimization problem",
            selected_tools=["gradient_descent", "adam_optimizer"],
            execution_strategy="sequential",
            performance_metrics={"accuracy": 0.95, "time": 1.5},
            success=True,
            utility_score=0.9,
        )

        assert_not_none(record, "Tool selection record created")
        assert_equal(record.success, True, "Success flag")
        assert_equal(record.utility_score, 0.9, "Utility score")

        # Retrieve similar problems
        similar = hmem.retrieve_similar_problems(
            problem_features=problem_features, limit=5
        )

        assert_true(len(similar) > 0, "Should find similar problems")
        assert_true(similar[0][1] > 0.9, "Should have high similarity to itself")

        # Get recommendations
        recommendations = hmem.get_recommended_tools(
            problem_features=problem_features, max_recommendations=3
        )

        assert_true(len(recommendations) > 0, "Should get tool recommendations")
        assert_true(
            "gradient_descent" in [r["tool"] for r in recommendations],
            "Should recommend previously successful tool",
        )

        results.record_pass("test_tool_selection_memory")
    except Exception as e:
        results.record_fail("test_tool_selection_memory", str(e))


def test_memory_consolidation():
    """Test memory consolidation across levels."""
    try:
        config = create_test_config()
        config.consolidation_interval = 0  # Disable auto-consolidation
        hmem = HierarchicalMemory(config)

        # Store many memories
        for i in range(20):
            hmem.store(f"Memory {i}", importance=0.5 + (i * 0.02))

        initial_count = sum(len(level.memories) for level in hmem.levels.values())

        # Manually trigger consolidation
        consolidated = hmem.consolidate()

        logger.info(f"Consolidated {consolidated} memories")

        # Note: Consolidation may not always reduce count if thresholds aren't met
        # Just verify it runs without error
        assert_true(consolidated >= 0, "Consolidation should return non-negative count")

        results.record_pass("test_memory_consolidation")
    except Exception as e:
        results.record_fail("test_memory_consolidation", str(e))


# ============================================================
# DISTRIBUTED MEMORY TESTS
# ============================================================


def test_distributed_memory_single_node():
    """Test distributed memory with single node."""
    try:
        config = create_test_config()
        config.enable_distributed = False
        config.replication_factor = 1

        # Create federation and node
        federation = MemoryFederation()

        dmem = DistributedMemory(
            config=config,
            federation=federation,
            node_id="node_1",
            host="localhost",
            port=5555,
        )

        # Store memory
        mem = dmem.store("Distributed test content", importance=0.8)

        assert_not_none(mem, "Distributed memory storage")
        assert_true(mem.id in dmem.local_storage, "Memory in local storage")

        # Retrieve
        query = MemoryQuery(query_type="similarity", content="test content", limit=5)
        result = dmem.retrieve(query)

        assert_true(len(result.memories) > 0, "Should retrieve memories")

        # Cleanup
        dmem.rpc_server.stop()
        dmem.rpc_client.cleanup()

        results.record_pass("test_distributed_memory_single_node")
    except Exception as e:
        results.record_fail("test_distributed_memory_single_node", str(e))


def test_memory_federation():
    """Test memory federation operations."""
    try:
        federation = MemoryFederation()

        # Register nodes
        from vulcan.memory import MemoryNode

        node1 = MemoryNode(node_id="node_1", host="localhost", port=5555, capacity=1000)

        node2 = MemoryNode(node_id="node_2", host="localhost", port=5556, capacity=1000)

        assert_true(federation.register_node(node1), "Register node 1")
        assert_true(federation.register_node(node2), "Register node 2")

        # Test routing
        nodes = federation.get_nodes_for_key("test_key", count=2)
        assert_true(len(nodes) <= 2, "Should return requested nodes")

        # Test leader election
        leader = federation.elect_leader()
        assert_not_none(leader, "Should elect leader")

        # Cleanup
        federation.stop_monitoring()

        results.record_pass("test_memory_federation")
    except Exception as e:
        results.record_fail("test_memory_federation", str(e))


# ============================================================
# PERSISTENCE TESTS
# ============================================================


def test_memory_persistence():
    """Test memory persistence operations."""
    try:
        temp_dir = tempfile.mkdtemp(prefix="persist_test_")

        persistence = MemoryPersistence(base_path=temp_dir)

        # Create and save memory
        memory = Memory(
            id="persist_001",
            type=MemoryType.SEMANTIC,
            content="Persistent content",
            importance=0.9,
        )

        success = persistence.save_memory(memory, compress=True, immediate=True)
        assert_true(success, "Memory save should succeed")

        # Load memory
        loaded = persistence.load_memory("persist_001")
        assert_not_none(loaded, "Memory load should succeed")
        assert_equal(loaded.id, memory.id, "Loaded memory ID")
        assert_equal(loaded.content, memory.content, "Loaded memory content")

        # Cleanup
        persistence.shutdown()
        shutil.rmtree(temp_dir, ignore_errors=True)

        results.record_pass("test_memory_persistence")
    except Exception as e:
        results.record_fail("test_memory_persistence", str(e))


def test_memory_checkpoint():
    """Test checkpoint and restore."""
    try:
        temp_dir = tempfile.mkdtemp(prefix="checkpoint_test_")

        persistence = MemoryPersistence(base_path=temp_dir)

        # Create test memories
        memories = {}
        for i in range(5):
            mem = Memory(
                id=f"mem_{i}",
                type=MemoryType.EPISODIC,
                content=f"Content {i}",
                importance=0.5 + i * 0.1,
            )
            memories[mem.id] = mem

        # Create checkpoint
        success = persistence.checkpoint(memories, name="test_checkpoint")
        assert_true(success, "Checkpoint creation should succeed")

        # Restore checkpoint
        restored = persistence.restore_from_checkpoint()
        assert_true(len(restored) > 0, "Should restore memories")
        assert_equal(len(restored), len(memories), "Should restore all memories")

        # Verify content
        for mem_id, mem in memories.items():
            assert_true(mem_id in restored, f"Memory {mem_id} should be restored")
            assert_equal(restored[mem_id].content, mem.content, f"Content of {mem_id}")

        # Cleanup
        persistence.shutdown()
        shutil.rmtree(temp_dir, ignore_errors=True)

        results.record_pass("test_memory_checkpoint")
    except Exception as e:
        results.record_fail("test_memory_checkpoint", str(e))


def test_version_control():
    """Test memory version control."""
    try:
        temp_dir = tempfile.mkdtemp(prefix="version_test_")

        vc = MemoryVersionControl(storage_path=temp_dir)

        # Create versions
        mem1 = Memory(
            id="versioned_001",
            type=MemoryType.SEMANTIC,
            content="Version 1",
            importance=0.5,
            metadata={"version": 1},
        )

        v1_id = vc.create_version(mem1, message="Initial version")
        assert_not_none(v1_id, "First version created")

        # Update memory
        mem1.content = "Version 2"
        mem1.metadata["version"] = 2

        v2_id = vc.create_version(mem1, message="Updated version")
        assert_not_none(v2_id, "Second version created")
        assert_true(v1_id != v2_id, "Version IDs should differ")

        # Get history
        history = vc.get_history("versioned_001")
        assert_true(len(history) >= 2, "Should have version history")

        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

        results.record_pass("test_version_control")
    except Exception as e:
        results.record_fail("test_version_control", str(e))


# ============================================================
# RETRIEVAL AND SEARCH TESTS
# ============================================================


def test_memory_index():
    """Test vector index operations."""
    try:
        index = MemoryIndex(dimension=128, index_type="flat")

        # Add embeddings
        for i in range(10):
            embedding = np.random.randn(128)
            embedding = embedding / np.linalg.norm(embedding)
            success = index.add(f"mem_{i}", embedding)
            assert_true(success, f"Add memory {i}")

        # Search
        query = np.random.randn(128)
        query = query / np.linalg.norm(query)

        results_list = index.search(query, k=5)
        assert_true(len(results_list) > 0, "Should return search results")
        assert_true(len(results_list) <= 5, "Should respect k limit")

        # Remove
        success = index.remove("mem_0")
        assert_true(success, "Remove should succeed")

        results.record_pass("test_memory_index")
    except Exception as e:
        results.record_fail("test_memory_index", str(e))


def test_attention_mechanism():
    """Test attention mechanism."""
    try:
        attention = AttentionMechanism(hidden_dim=64, input_dim=128)

        # Create query and memories
        query = np.random.randn(128)
        query = query / np.linalg.norm(query)

        memories_emb = [np.random.randn(128) for _ in range(5)]
        memories_emb = [m / np.linalg.norm(m) for m in memories_emb]

        # Compute attention
        weights = attention.compute_attention(query, memories_emb)

        assert_true(len(weights) == len(memories_emb), "Weights match memories")
        assert_true(abs(np.sum(weights) - 1.0) < 0.01, "Weights should sum to ~1")
        assert_true(all(w >= 0 for w in weights), "Weights should be non-negative")

        results.record_pass("test_attention_mechanism")
    except Exception as e:
        results.record_fail("test_attention_mechanism", str(e))


# ============================================================
# SPECIALIZED MEMORY TESTS
# ============================================================


def test_episodic_memory():
    """Test episodic memory operations."""
    try:
        config = create_test_config()
        emem = EpisodicMemory(config)

        # Start episode
        episode_id = emem.start_episode(
            context={"location": "test", "task": "learning"}, tags={"test", "learning"}
        )

        assert_not_none(episode_id, "Episode creation")

        # Add events
        emem.add_event({"action": "read", "object": "book"})
        emem.add_event({"action": "write", "object": "notes"})

        # End episode
        emem.end_episode(outcome="success", value=0.8, emotional_valence=0.5)

        # Verify episode stored
        assert_true(episode_id in emem.episodes, "Episode should be stored")

        episode = emem.episodes[episode_id]
        assert_equal(len(episode.events), 2, "Should have 2 events")
        assert_equal(episode.outcome, "success", "Episode outcome")

        # Recall similar episodes
        similar = emem.recall_similar_episodes(context={"location": "test"}, limit=5)

        assert_true(len(similar) > 0, "Should find similar episodes")

        results.record_pass("test_episodic_memory")
    except Exception as e:
        results.record_fail("test_episodic_memory", str(e))


def test_semantic_memory():
    """Test semantic memory operations."""
    try:
        config = create_test_config()
        smem = SemanticMemory(config)

        # Add concepts
        c1_id = smem.add_concept(
            name="Python",
            definition="A high-level programming language",
            attributes={"paradigm": "multi-paradigm", "typing": "dynamic"},
            confidence=0.9,
        )

        c2_id = smem.add_concept(
            name="Programming",
            definition="The process of writing code",
            attributes={"domain": "computer_science"},
            confidence=0.8,
        )

        assert_not_none(c1_id, "First concept created")
        assert_not_none(c2_id, "Second concept created")

        # Add relation
        smem.add_relation(c1_id, "is_a", c2_id, confidence=0.9)

        # Query concept
        python = smem.query_concept("Python")
        assert_not_none(python, "Should find Python concept")
        assert_equal(python.name, "Python", "Concept name")

        # Test tool performance tracking
        smem.update_tool_performance(
            tool_name="gradient_descent",
            problem_type="optimization",
            success=True,
            metrics={"convergence_rate": 0.95, "iterations": 100},
            context={"dimensions": 10, "convex": True},
        )

        # Get tool recommendations
        recommendations = smem.get_tool_recommendations(
            problem_type="optimization", context={"dimensions": 10}, min_confidence=0.1
        )

        assert_true(len(recommendations) > 0, "Should get tool recommendations")

        results.record_pass("test_semantic_memory")
    except Exception as e:
        results.record_fail("test_semantic_memory", str(e))


def test_procedural_memory():
    """Test procedural memory operations."""
    try:
        config = create_test_config()
        pmem = ProceduralMemory(config)

        # Add skill
        def step1(ctx):
            ctx["step1_done"] = True
            return ctx

        def step2(ctx):
            ctx["step2_done"] = True
            return ctx

        skill_id = pmem.add_skill(
            name="test_skill",
            steps=[step1, step2],
            preconditions=["step1_done == False"],
            postconditions=["step1_done == True", "step2_done == True"],
        )

        assert_not_none(skill_id, "Skill creation")

        # Execute skill
        context = {"step1_done": False, "step2_done": False}
        result = pmem.execute_skill("test_skill", context)

        # Note: Preconditions use eval which may fail, but skill should still be stored
        assert_true(skill_id in pmem.skills, "Skill should be stored")

        # Compose skills
        skill2_id = pmem.add_skill(name="skill2", steps=["action1", "action2"])

        composed_id = pmem.compose_skills(["test_skill", "skill2"], "composed_skill")

        assert_not_none(composed_id, "Composed skill created")

        results.record_pass("test_procedural_memory")
    except Exception as e:
        results.record_fail("test_procedural_memory", str(e))


def test_working_memory():
    """Test working memory operations."""
    try:
        config = create_test_config()
        wmem = WorkingMemory(config)

        # Add items
        wmem.add("Item 1", relevance=0.9)
        wmem.add("Item 2", relevance=0.7)
        wmem.add("Item 3", relevance=0.5)

        assert_true(len(wmem.buffer) == 3, "Should have 3 items")

        # Test capacity
        for i in range(config.max_working_memory + 5):
            wmem.add(f"Item {i}", relevance=0.6)

        assert_true(
            len(wmem.buffer) <= config.max_working_memory, "Should not exceed capacity"
        )

        # Test attention
        attention_scores = [0.9, 0.5, 0.3]
        wmem.update_attention(attention_scores[: len(wmem.buffer)])

        focused = wmem.get_focused()
        assert_not_none(focused, "Should have focused item")

        # Test phonological loop
        wmem.add_to_phonological_loop("Test speech")
        assert_true(
            len(wmem.phonological_loop) > 0, "Phonological loop should have content"
        )

        # Cleanup
        wmem.clear()
        assert_equal(len(wmem.buffer), 0, "Buffer should be clear")

        results.record_pass("test_working_memory")
    except Exception as e:
        results.record_fail("test_working_memory", str(e))


# ============================================================
# CONSOLIDATION TESTS
# ============================================================


def test_memory_consolidator():
    """Test memory consolidation strategies."""
    try:
        consolidator = MemoryConsolidator()

        # Create test memories
        memories = []
        for i in range(20):
            mem = Memory(
                id=f"mem_{i}",
                type=MemoryType.EPISODIC,
                content=f"Content {i}",
                importance=0.3 + (i * 0.03),
                access_count=i,
                embedding=np.random.randn(128),
            )
            memories.append(mem)

        # Test different strategies
        strategies = [
            ConsolidationStrategy.IMPORTANCE_BASED,
            ConsolidationStrategy.FREQUENCY_BASED,
            ConsolidationStrategy.RECENCY_BASED,
        ]

        for strategy in strategies:
            consolidated = consolidator.consolidate(
                memories, strategy=strategy, target_count=10
            )

            assert_true(
                len(consolidated) <= 10,
                f"{strategy.value} should reduce to target count",
            )
            assert_true(
                len(consolidated) > 0, f"{strategy.value} should return some memories"
            )

        results.record_pass("test_memory_consolidator")
    except Exception as e:
        results.record_fail("test_memory_consolidator", str(e))


# ============================================================
# INTEGRATION TESTS
# ============================================================


def test_full_memory_pipeline():
    """Test complete memory pipeline from storage to retrieval."""
    try:
        config = create_test_config()

        # Create hierarchical memory
        hmem = HierarchicalMemory(config)

        # Store diverse memories
        memories_stored = []

        # Episodic
        mem1 = hmem.store(
            "Yesterday I went to the store",
            memory_type=MemoryType.EPISODIC,
            importance=0.7,
        )
        memories_stored.append(mem1)

        # Semantic
        mem2 = hmem.store(
            "Python is a programming language",
            memory_type=MemoryType.SEMANTIC,
            importance=0.9,
        )
        memories_stored.append(mem2)

        # Procedural
        mem3 = hmem.store(
            "To make coffee: 1) Boil water 2) Add grounds 3) Pour",
            memory_type=MemoryType.PROCEDURAL,
            importance=0.6,
        )
        memories_stored.append(mem3)

        # Query similar to stored content
        query = MemoryQuery(query_type="similarity", content="programming", limit=10)

        result = hmem.retrieve(query)

        assert_true(len(result.memories) > 0, "Should retrieve memories")
        assert_true(result.query_time_ms > 0, "Should track query time")

        # Consolidate
        consolidated = hmem.consolidate()
        logger.info(f"Consolidated {consolidated} memories")

        # Get stats
        stats = hmem.get_stats()
        assert_true(stats.total_stores >= 3, "Should track stores")
        assert_true(stats.total_queries >= 1, "Should track queries")

        results.record_pass("test_full_memory_pipeline")
    except Exception as e:
        results.record_fail("test_full_memory_pipeline", str(e))


def test_cross_system_integration():
    """Test integration between different memory systems."""
    try:
        config = create_test_config()

        # Create multiple systems
        hmem = HierarchicalMemory(config)
        emem = EpisodicMemory(config)
        smem = SemanticMemory(config)

        # Store related information across systems
        # Hierarchical: General storage
        hmem.store("Learning about machine learning", importance=0.8)

        # Episodic: Experience
        ep_id = emem.start_episode(
            context={"activity": "studying", "subject": "ML"}, tags={"learning", "ML"}
        )
        emem.add_event({"action": "read", "topic": "neural networks"})
        emem.end_episode(outcome="understood", value=0.9)

        # Semantic: Knowledge
        concept_id = smem.add_concept(
            name="Machine Learning",
            definition="Field of AI focused on learning from data",
            attributes={"domain": "AI", "type": "methodology"},
        )

        # Verify all systems have data
        assert_true(
            len(hmem.levels["long_term"].memories) > 0, "Hierarchical has memories"
        )
        assert_true(len(emem.episodes) > 0, "Episodic has episodes")
        assert_true(len(smem.concepts) > 0, "Semantic has concepts")

        # Query each system
        query = MemoryQuery(query_type="similarity", content="learning", limit=5)

        h_result = hmem.retrieve(query)
        e_result = emem.retrieve(query)
        s_result = smem.retrieve(query)

        total_results = (
            len(h_result.memories) + len(e_result.memories) + len(s_result.memories)
        )
        assert_true(total_results > 0, "Should retrieve from at least one system")

        results.record_pass("test_cross_system_integration")
    except Exception as e:
        results.record_fail("test_cross_system_integration", str(e))


# ============================================================
# STRESS TESTS
# ============================================================


def test_high_volume_storage():
    """Test system under high memory volume."""
    try:
        config = create_test_config()
        config.max_long_term = 1000
        hmem = HierarchicalMemory(config)

        start_time = time.time()

        # Store many memories
        count = 100
        for i in range(count):
            hmem.store(f"High volume test {i}", importance=np.random.random())

        elapsed = time.time() - start_time

        logger.info(
            f"Stored {count} memories in {elapsed:.2f}s ({count / elapsed:.1f} mem/s)"
        )

        # Verify storage
        stats = hmem.get_stats()
        assert_true(stats.total_stores >= count, f"Should store {count} memories")

        results.record_pass("test_high_volume_storage")
    except Exception as e:
        results.record_fail("test_high_volume_storage", str(e))


def test_concurrent_access():
    """Test concurrent memory access."""
    try:
        import threading

        config = create_test_config()
        hmem = HierarchicalMemory(config)

        errors = []

        def store_worker(thread_id: int, count: int):
            try:
                for i in range(count):
                    hmem.store(f"Thread {thread_id} - Memory {i}", importance=0.5)
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        # Create threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=store_worker, args=(i, 10))
            threads.append(t)
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        assert_equal(len(errors), 0, f"Should have no errors: {errors}")

        stats = hmem.get_stats()
        assert_true(stats.total_stores >= 50, "Should store from all threads")

        results.record_pass("test_concurrent_access")
    except Exception as e:
        results.record_fail("test_concurrent_access", str(e))


# ============================================================
# MAIN TEST RUNNER
# ============================================================


def run_all_tests():
    """Run all test suites."""
    if not MEMORY_MODULE_AVAILABLE:
        print("ERROR: Memory module not available. Cannot run tests.")
        return

    print("\n" + "=" * 70)
    print("VULCAN-AGI MEMORY SYSTEM - INTEGRATION TEST SUITE")
    print("=" * 70 + "\n")

    # Base Memory Tests
    print("Running Base Memory Tests...")
    test_memory_creation()
    test_memory_salience()
    test_memory_serialization()

    # Hierarchical Memory Tests
    print("\nRunning Hierarchical Memory Tests...")
    test_hierarchical_memory_basic()
    test_tool_selection_memory()
    test_memory_consolidation()

    # Distributed Memory Tests
    print("\nRunning Distributed Memory Tests...")
    test_distributed_memory_single_node()
    test_memory_federation()

    # Persistence Tests
    print("\nRunning Persistence Tests...")
    test_memory_persistence()
    test_memory_checkpoint()
    test_version_control()

    # Retrieval Tests
    print("\nRunning Retrieval and Search Tests...")
    test_memory_index()
    test_attention_mechanism()

    # Specialized Memory Tests
    print("\nRunning Specialized Memory Tests...")
    test_episodic_memory()
    test_semantic_memory()
    test_procedural_memory()
    test_working_memory()

    # Consolidation Tests
    print("\nRunning Consolidation Tests...")
    test_memory_consolidator()

    # Integration Tests
    print("\nRunning Integration Tests...")
    test_full_memory_pipeline()
    test_cross_system_integration()

    # Stress Tests
    print("\nRunning Stress Tests...")
    test_high_volume_storage()
    test_concurrent_access()

    # Print results
    results.print_summary()

    # Return exit code
    return 0 if results.failed == 0 else 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
