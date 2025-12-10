"""
Comprehensive test suite for specialized.py

Tests cover:
- Episodic Memory (episodes, events, patterns)
- Semantic Memory (concepts, relations, tool performance)
- Procedural Memory (skills, execution, composition)
- Working Memory (buffers, attention, rehearsal)
- All memory consolidation and retrieval operations
- Edge cases and error handling
"""

from vulcan.memory.specialized import (Concept, Episode, EpisodicMemory,
                                       ProceduralMemory, SemanticMemory,
                                       Skill, ToolPerformanceConcept,
                                       WorkingMemory)
from vulcan.memory.base import (Memory, MemoryConfig, MemoryQuery, MemoryType,
                                RetrievalResult)
import shutil
# Import the module under test
import sys
import tempfile
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# FIXTURES
# ============================================================


@pytest.fixture
def memory_config():
    """Create test memory configuration."""
    return MemoryConfig(
        max_working_memory=20,
        max_short_term=1000,
        max_long_term=100000,
        consolidation_interval=1000,
    )


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_episode():
    """Create sample episode."""
    return Episode(
        id="test_episode_001",
        start_time=time.time() - 3600,
        end_time=time.time(),
        events=[
            {
                "type": "action",
                "content": "started task",
                "timestamp": time.time() - 3600,
            },
            {
                "type": "observation",
                "content": "observed result",
                "timestamp": time.time() - 1800,
            },
        ],
        context={"task": "test", "environment": "dev"},
        outcome="success",
        value=0.8,
        emotional_valence=0.6,
        tags={"test", "development"},
        importance=0.7,
    )


@pytest.fixture
def sample_concept():
    """Create sample concept."""
    return Concept(
        id="concept_001",
        name="Machine Learning",
        definition="A field of AI that uses statistical techniques to give computer systems the ability to learn",
        attributes={"domain": "AI", "complexity": "high"},
        relationships={"is_a": ["artificial_intelligence"], "uses": ["statistics"]},
        confidence=0.8,
    )


@pytest.fixture
def sample_tool_concept():
    """Create sample tool performance concept."""
    return ToolPerformanceConcept(
        id="tool_001",
        name="probabilistic_reasoning",
        definition="Statistical inference tool",
        tool_type="probabilistic",
        attributes={
            "strengths": ["uncertainty handling"],
            "weaknesses": ["computational cost"],
        },
        relationships={},
        performance_metrics={"accuracy": 0.85, "speed": 0.7},
        problem_domains=["prediction", "classification"],
        confidence=0.75,
    )


@pytest.fixture
def sample_skill():
    """Create sample skill."""
    return Skill(
        id="skill_001",
        name="data_preprocessing",
        steps=["load_data", "clean_data", "normalize_data"],
        preconditions=["data_exists"],
        postconditions=["data_is_clean"],
        performance_history=[1.0, 1.0, 0.5, 1.0],
        success_rate=0.875,
        execution_time=2.5,
        dependencies=[],
    )


# ============================================================
# EPISODE TESTS
# ============================================================


class TestEpisode:
    """Test Episode class."""

    def test_create_episode(self):
        """Test creating an episode."""
        episode = Episode(
            id="test_001",
            start_time=time.time(),
            end_time=None,
            events=[],
            context={"key": "value"},
        )

        assert episode.id == "test_001"
        assert episode.context == {"key": "value"}
        assert episode.end_time is None
        assert len(episode.events) == 0

    def test_add_event(self, sample_episode):
        """Test adding event to episode."""
        initial_count = len(sample_episode.events)

        sample_episode.add_event({"type": "new_event", "content": "test"})

        assert len(sample_episode.events) == initial_count + 1
        assert sample_episode.events[-1]["content"] == "test"
        assert "timestamp" in sample_episode.events[-1]
        assert "sequence" in sample_episode.events[-1]

    def test_compute_duration(self, sample_episode):
        """Test computing episode duration."""
        duration = sample_episode.compute_duration()

        assert duration > 0
        # Allow small tolerance for floating-point precision in time calculations
        assert (
            duration <= 3600.001
        )  # Should be at most 1 hour (with floating-point tolerance)

    def test_get_summary(self, sample_episode):
        """Test getting episode summary."""
        summary = sample_episode.get_summary()

        assert "id" in summary
        assert "duration" in summary
        assert "num_events" in summary
        assert "outcome" in summary
        assert "value" in summary
        assert summary["id"] == sample_episode.id
        assert summary["num_events"] == len(sample_episode.events)

    def test_to_memory(self, sample_episode):
        """Test converting episode to Memory."""
        memory = sample_episode.to_memory()

        assert isinstance(memory, Memory)
        assert memory.type == MemoryType.EPISODIC
        assert memory.id == sample_episode.id
        assert "events" in memory.content
        assert memory.importance == sample_episode.importance


# ============================================================
# EPISODIC MEMORY TESTS
# ============================================================


class TestEpisodicMemory:
    """Test EpisodicMemory class."""

    def test_create_episodic_memory(self, memory_config):
        """Test creating episodic memory system."""
        em = EpisodicMemory(memory_config)

        assert em is not None
        assert len(em.episodes) == 0
        assert em.current_episode is None

    def test_start_episode(self, memory_config):
        """Test starting a new episode."""
        em = EpisodicMemory(memory_config)

        context = {"task": "test", "user": "alice"}
        tags = {"test", "example"}

        episode_id = em.start_episode(context, tags)

        assert episode_id is not None
        assert em.current_episode is not None
        assert em.current_episode.id == episode_id
        assert em.current_episode.context == context
        assert em.current_episode.tags == tags
        assert episode_id in em.episodes

    def test_end_episode(self, memory_config):
        """Test ending an episode."""
        em = EpisodicMemory(memory_config)

        em.start_episode({"task": "test"})
        em.add_event({"type": "action", "content": "did something"})

        em.end_episode(outcome="success", value=0.9, emotional_valence=0.7)

        assert em.current_episode is None
        assert len(em.episodes) == 1

        episode = list(em.episodes.values())[0]
        assert episode.outcome == "success"
        assert episode.value == 0.9
        assert episode.emotional_valence == 0.7
        assert episode.end_time is not None

    def test_add_event(self, memory_config):
        """Test adding event to current episode."""
        em = EpisodicMemory(memory_config)

        em.start_episode({"task": "test"})

        em.add_event({"type": "action", "content": "step 1"})
        em.add_event({"type": "observation", "content": "result"})

        assert len(em.current_episode.events) == 2
        assert em.current_episode.events[0]["content"] == "step 1"
        assert em.current_episode.events[1]["content"] == "result"

    def test_auto_end_long_episode(self, memory_config):
        """Test auto-ending of long episodes."""
        em = EpisodicMemory(memory_config)

        em.start_episode({"task": "test"})

        # Add many events to trigger auto-end
        for i in range(101):
            em.add_event({"type": "action", "content": f"event {i}"})

        # Should have auto-ended and started new episode or have no current
        # (behavior may vary based on implementation)
        assert em.current_episode is None or len(em.current_episode.events) < 101

    def test_store_episodic_memory(self, memory_config):
        """Test storing episodic memory."""
        em = EpisodicMemory(memory_config)

        content = {"description": "test memory", "data": "example"}
        memory = em.store(content, importance=0.7)

        assert isinstance(memory, Memory)
        assert memory.type == MemoryType.EPISODIC
        assert len(em.episodes) >= 1

    def test_recall_similar_episodes(self, memory_config):
        """Test recalling similar episodes."""
        em = EpisodicMemory(memory_config)

        # Create several episodes with different contexts
        em.start_episode({"task": "coding", "language": "python"})
        em.end_episode()

        em.start_episode({"task": "coding", "language": "java"})
        em.end_episode()

        em.start_episode({"task": "writing", "type": "documentation"})
        em.end_episode()

        # Recall episodes similar to coding context
        similar = em.recall_similar_episodes({"task": "coding"}, limit=5)

        assert len(similar) >= 2
        # Should prefer coding episodes
        coding_episodes = [item for item in similar if item.context.get("task") == "coding"]
        assert len(coding_episodes) >= 2

    def test_recall_by_tags(self, memory_config):
        """Test recalling episodes by tags."""
        em = EpisodicMemory(memory_config)

        em.start_episode({"task": "test1"}, tags={"development", "testing"})
        em.end_episode()

        em.start_episode({"task": "test2"}, tags={"development", "production"})
        em.end_episode()

        em.start_episode({"task": "test3"}, tags={"research"})
        em.end_episode()

        # Recall by development tag
        dev_episodes = em.recall_by_tags({"development"}, limit=10)

        assert len(dev_episodes) == 2
        for ep in dev_episodes:
            assert "development" in ep.tags

    def test_recall_by_value_range(self, memory_config):
        """Test recalling episodes by value range."""
        em = EpisodicMemory(memory_config)

        em.start_episode({"task": "low"})
        em.end_episode(value=0.2)

        em.start_episode({"task": "medium"})
        em.end_episode(value=0.5)

        em.start_episode({"task": "high"})
        em.end_episode(value=0.9)

        # Recall high-value episodes
        high_value = em.recall_by_value_range(0.7, 1.0)

        assert len(high_value) == 1
        assert high_value[0].value >= 0.7

    def test_consolidate_episodes(self, memory_config):
        """Test episode consolidation."""
        em = EpisodicMemory(memory_config)

        # Create old similar episodes
        old_time = time.time() - (8 * 24 * 3600)  # 8 days ago

        for i in range(3):
            episode = Episode(
                id=f"old_{i}",
                start_time=old_time - i * 100,
                end_time=old_time - i * 100 + 50,
                events=[{"type": "test"}],
                context={"similar": True, "index": i},
            )
            em.episodes[episode.id] = episode
            em._index_episode(episode)

        initial_count = len(em.episodes)
        consolidated = em.consolidate()

        # Should have consolidated some episodes
        assert consolidated >= 0
        assert len(em.episodes) <= initial_count

    def test_retrieve_episodic_memory(self, memory_config):
        """Test retrieving episodic memories."""
        em = EpisodicMemory(memory_config)

        # Store some memories
        em.store({"content": "test memory 1"})
        em.store({"content": "test memory 2"})

        # Query memories - FIXED: Use keyword arguments
        query = MemoryQuery(
            query_type="episodic",
            content={"context": {"task": "test"}},
            embedding=None,
            time_range=None,
            filters={},
            limit=5,
        )

        result = em.retrieve(query)

        assert isinstance(result, RetrievalResult)
        assert len(result.memories) >= 0

    def test_forget_episode(self, memory_config):
        """Test forgetting an episode."""
        em = EpisodicMemory(memory_config)

        em.start_episode({"task": "test"})
        episode_id = em.current_episode.id
        em.end_episode()

        assert episode_id in em.episodes

        success = em.forget(episode_id)

        assert success is True
        assert episode_id not in em.episodes


# ============================================================
# CONCEPT TESTS
# ============================================================


class TestConcept:
    """Test Concept class."""

    def test_create_concept(self):
        """Test creating a concept."""
        concept = Concept(
            id="test_001",
            name="Test Concept",
            definition="A concept for testing",
            attributes={"type": "test"},
            relationships={"related_to": ["other_concept"]},
        )

        assert concept.id == "test_001"
        assert concept.name == "Test Concept"
        assert concept.confidence == 0.5  # default
        assert concept.frequency == 1

    def test_tool_performance_concept(self):
        """Test ToolPerformanceConcept."""
        tool_concept = ToolPerformanceConcept(
            id="tool_001",
            name="test_tool",
            definition="A test tool",
            tool_type="reasoning",
            attributes={},
            relationships={},
            performance_metrics={"accuracy": 0.8},
            problem_domains=["classification"],
        )

        assert isinstance(tool_concept, Concept)
        assert tool_concept.tool_type == "reasoning"
        assert "accuracy" in tool_concept.performance_metrics
        assert "classification" in tool_concept.problem_domains


# ============================================================
# SEMANTIC MEMORY TESTS
# ============================================================


class TestSemanticMemory:
    """Test SemanticMemory class."""

    def test_create_semantic_memory(self, memory_config):
        """Test creating semantic memory system."""
        sm = SemanticMemory(memory_config)

        assert sm is not None
        assert len(sm.concepts) >= 5  # Should have initial tool concepts

    def test_add_concept(self, memory_config):
        """Test adding a concept."""
        sm = SemanticMemory(memory_config)

        concept_id = sm.add_concept(
            name="Neural Network",
            definition="A computational model inspired by biological neural networks",
            attributes={"domain": "machine learning"},
            confidence=0.8,
        )

        assert concept_id is not None
        assert concept_id in sm.concepts
        assert sm.concepts[concept_id].name == "Neural Network"
        assert "Neural Network" in sm.name_index

    def test_add_duplicate_concept(self, memory_config):
        """Test adding duplicate concept increases frequency."""
        sm = SemanticMemory(memory_config)

        concept_id1 = sm.add_concept("Test", "Definition 1")
        concept_id2 = sm.add_concept("Test", "Definition 2")

        assert concept_id1 == concept_id2
        assert sm.concepts[concept_id1].frequency == 2

    def test_add_relation(self, memory_config):
        """Test adding relation between concepts."""
        sm = SemanticMemory(memory_config)

        c1_id = sm.add_concept("Concept A", "First concept")
        c2_id = sm.add_concept("Concept B", "Second concept")

        sm.add_relation(c1_id, "is_related_to", c2_id, confidence=0.9)

        assert "is_related_to" in sm.concepts[c1_id].relationships
        assert c2_id in sm.concepts[c1_id].relationships["is_related_to"]

    def test_query_concept(self, memory_config):
        """Test querying concept by name."""
        sm = SemanticMemory(memory_config)

        sm.add_concept("Query Test", "A concept to query")

        concept = sm.query_concept("Query Test")

        assert concept is not None
        assert concept.name == "Query Test"

    def test_add_tool_performance_concept(self, memory_config):
        """Test adding tool performance concept."""
        sm = SemanticMemory(memory_config)

        initial_count = len(sm.concepts)

        tool_id = sm.add_tool_performance_concept(
            name="test_tool",
            definition="A tool for testing",
            tool_type="testing",
            attributes={"speed": "fast"},
            problem_domains=["unit_testing", "integration_testing"],
            performance_metrics={"accuracy": 0.95, "speed": 0.8},
            confidence=0.85,
        )

        assert tool_id is not None
        assert len(sm.concepts) > initial_count

        concept = sm.concepts[tool_id]
        assert isinstance(concept, ToolPerformanceConcept)
        assert concept.tool_type == "testing"
        assert "accuracy" in concept.performance_metrics

    def test_update_tool_performance(self, memory_config):
        """Test updating tool performance."""
        sm = SemanticMemory(memory_config)

        # Add a tool
        sm.add_tool_performance_concept(
            name="updatable_tool",
            definition="Tool to update",
            tool_type="test",
            performance_metrics={"accuracy": 0.5},
        )

        # Update performance
        sm.update_tool_performance(
            tool_name="updatable_tool",
            problem_type="test",
            success=True,
            metrics={"accuracy": 0.9, "precision": 0.85},
            context={"complexity": "low"},
        )

        concept_id = sm.name_index["updatable_tool"]
        concept = sm.concepts[concept_id]

        assert "accuracy" in concept.performance_metrics
        assert "precision" in concept.performance_metrics
        assert len(concept.success_contexts) == 1

    def test_get_tool_recommendations(self, memory_config):
        """Test getting tool recommendations."""
        sm = SemanticMemory(memory_config)

        # The system should have initial tool concepts
        recommendations = sm.get_tool_recommendations(
            problem_type="prediction",
            context={"data_type": "numerical"},
            min_confidence=0.3,
        )

        assert isinstance(recommendations, list)
        # Should have at least probabilistic reasoning for prediction
        assert len(recommendations) >= 1

        for rec in recommendations:
            assert "tool_name" in rec
            assert "confidence" in rec
            assert "suitability" in rec

    def test_find_tool_relationships(self, memory_config):
        """Test finding relationships between tools."""
        sm = SemanticMemory(memory_config)

        # Use an existing tool
        relationships = sm.find_tool_relationships("probabilistic_reasoning")

        assert isinstance(relationships, dict)
        assert "complements" in relationships
        assert "substitutes" in relationships

    def test_store_semantic_memory(self, memory_config):
        """Test storing semantic memory."""
        sm = SemanticMemory(memory_config)

        memory = sm.store(
            {
                "name": "Test Semantic",
                "definition": "A test semantic memory",
                "attributes": {"category": "test"},
            }
        )

        assert isinstance(memory, Memory)
        assert memory.type == MemoryType.SEMANTIC

    def test_retrieve_semantic_memory(self, memory_config):
        """Test retrieving semantic memories."""
        sm = SemanticMemory(memory_config)

        # Add some concepts
        sm.add_concept("Retrieval Test 1", "First test")
        sm.add_concept("Retrieval Test 2", "Second test")

        # FIXED: Use keyword arguments
        query = MemoryQuery(
            query_type="semantic",
            content="Retrieval Test 1",
            embedding=None,
            time_range=None,
            filters={},
            limit=5,
        )

        result = sm.retrieve(query)

        assert isinstance(result, RetrievalResult)
        assert len(result.memories) >= 1

    def test_forget_concept(self, memory_config):
        """Test forgetting a concept."""
        sm = SemanticMemory(memory_config)

        concept_id = sm.add_concept("Forgettable", "Will be forgotten")

        assert concept_id in sm.concepts

        success = sm.forget(concept_id)

        assert success is True
        assert concept_id not in sm.concepts

    def test_consolidate_concepts(self, memory_config):
        """Test concept consolidation."""
        sm = SemanticMemory(memory_config)

        # Add similar concepts
        sm.add_concept("Similar One", "Definition")
        sm.add_concept("Similar Two", "Definition")

        initial_count = len(sm.concepts)
        consolidated = sm.consolidate()

        assert consolidated >= 0
        assert len(sm.concepts) <= initial_count

    def test_infer_relations(self, memory_config):
        """Test inferring relations."""
        sm = SemanticMemory(memory_config)

        # Create chain of concepts
        c1 = sm.add_concept("A", "First")
        c2 = sm.add_concept("B", "Second")
        c3 = sm.add_concept("C", "Third")

        sm.add_relation(c1, "leads_to", c2)
        sm.add_relation(c2, "leads_to", c3)

        # Infer transitive relations
        inferred = sm.infer_relations(c1, max_depth=2)

        assert isinstance(inferred, dict)


# ============================================================
# SKILL TESTS
# ============================================================


class TestSkill:
    """Test Skill class."""

    def test_create_skill(self):
        """Test creating a skill."""
        skill = Skill(
            id="skill_001",
            name="test_skill",
            steps=["step1", "step2"],
            preconditions=["precond1"],
            postconditions=["postcond1"],
            performance_history=[],
        )

        assert skill.id == "skill_001"
        assert skill.name == "test_skill"
        assert len(skill.steps) == 2
        assert skill.success_rate == 0.0

    def test_execute_skill_with_callable(self):
        """Test executing skill with callable steps."""

        def step1(context):
            context["step1_done"] = True
            return context

        def step2(context):
            context["step2_done"] = True
            return context

        skill = Skill(
            id="skill_001",
            name="callable_skill",
            steps=[step1, step2],
            preconditions=[],
            postconditions=[],
            performance_history=[],
        )

        result = skill.execute({})

        assert result["step1_done"] is True
        assert result["step2_done"] is True

    def test_update_performance(self):
        """Test updating skill performance."""
        skill = Skill(
            id="skill_001",
            name="test",
            steps=[],
            preconditions=[],
            postconditions=[],
            performance_history=[],
        )

        skill.update_performance(success=True, execution_time=1.5)
        skill.update_performance(success=True, execution_time=2.0)
        skill.update_performance(success=False, execution_time=1.0)

        assert len(skill.performance_history) == 3
        assert skill.success_rate > 0
        assert skill.success_rate < 1


# ============================================================
# PROCEDURAL MEMORY TESTS
# ============================================================


class TestProceduralMemory:
    """Test ProceduralMemory class."""

    def test_create_procedural_memory(self, memory_config):
        """Test creating procedural memory system."""
        pm = ProceduralMemory(memory_config)

        assert pm is not None
        assert len(pm.skills) == 0

    def test_add_skill(self, memory_config):
        """Test adding a skill."""
        pm = ProceduralMemory(memory_config)

        skill_id = pm.add_skill(
            name="test_skill",
            steps=["step1", "step2", "step3"],
            preconditions=["input_ready"],
            postconditions=["output_ready"],
        )

        assert skill_id is not None
        assert skill_id in pm.skills
        assert pm.skills[skill_id].name == "test_skill"

    def test_find_skill(self, memory_config):
        """Test finding skill by name."""
        pm = ProceduralMemory(memory_config)

        pm.add_skill("findable_skill", ["step1"])

        skill = pm.find_skill("findable_skill")

        assert skill is not None
        assert skill.name == "findable_skill"

    def test_execute_skill(self, memory_config):
        """Test executing a skill."""
        pm = ProceduralMemory(memory_config)

        def process(context):
            context["processed"] = True
            return context

        pm.add_skill(
            name="executable_skill",
            steps=[process],
            preconditions=[],
            postconditions=[],
        )

        result = pm.execute_skill("executable_skill", {})

        assert result is not None
        assert result.get("processed") is True

    def test_compose_skills(self, memory_config):
        """Test composing multiple skills."""
        pm = ProceduralMemory(memory_config)

        pm.add_skill("skill_a", ["step_a1", "step_a2"])
        pm.add_skill("skill_b", ["step_b1", "step_b2"])

        composed_id = pm.compose_skills(["skill_a", "skill_b"], "composed_skill")

        assert composed_id is not None
        assert composed_id in pm.skills

        composed = pm.skills[composed_id]
        assert len(composed.steps) == 4  # Combined steps

    def test_store_procedural_memory(self, memory_config):
        """Test storing procedural memory."""
        pm = ProceduralMemory(memory_config)

        memory = pm.store({"name": "stored_skill", "steps": ["step1", "step2"]})

        assert isinstance(memory, Memory)
        assert memory.type == MemoryType.PROCEDURAL

    def test_retrieve_procedural_memory(self, memory_config):
        """Test retrieving procedural memories."""
        pm = ProceduralMemory(memory_config)

        pm.add_skill("retrieval_test", ["step1"])

        # FIXED: Use keyword arguments
        query = MemoryQuery(
            query_type="procedural",
            content="retrieval_test",
            embedding=None,
            time_range=None,
            filters={},
            limit=5,
        )

        result = pm.retrieve(query)

        assert isinstance(result, RetrievalResult)
        assert len(result.memories) >= 1

    def test_forget_skill(self, memory_config):
        """Test forgetting a skill."""
        pm = ProceduralMemory(memory_config)

        skill_id = pm.add_skill("forgettable_skill", ["step"])

        assert skill_id in pm.skills

        success = pm.forget(skill_id)

        assert success is True
        assert skill_id not in pm.skills

    def test_consolidate_skills(self, memory_config):
        """Test skill consolidation."""
        pm = ProceduralMemory(memory_config)

        # Add low-performing skill
        skill_id = pm.add_skill("low_performer", ["step"])
        skill = pm.skills[skill_id]

        # Simulate poor performance
        for _ in range(15):
            skill.update_performance(success=False, execution_time=1.0)

        len(pm.skills)
        consolidated = pm.consolidate()

        # Should remove low-performing skills
        assert consolidated >= 0


# ============================================================
# WORKING MEMORY TESTS
# ============================================================


class TestWorkingMemory:
    """Test WorkingMemory class."""

    def test_create_working_memory(self, memory_config):
        """Test creating working memory system."""
        wm = WorkingMemory(memory_config)

        assert wm is not None
        assert wm.capacity == memory_config.max_working_memory
        assert len(wm.buffer) == 0

    def test_add_to_working_memory(self, memory_config):
        """Test adding items to working memory."""
        wm = WorkingMemory(memory_config)

        success = wm.add("test content", relevance=0.8)

        assert success is True
        assert len(wm.buffer) == 1

    def test_capacity_limit(self, memory_config):
        """Test working memory capacity limit."""
        wm = WorkingMemory(memory_config)

        # Fill beyond capacity
        for i in range(wm.capacity + 5):
            wm.add(f"content {i}", relevance=0.5)

        # Should not exceed capacity
        assert len(wm.buffer) <= wm.capacity

    def test_focus_high_relevance(self, memory_config):
        """Test that high relevance items become focus."""
        wm = WorkingMemory(memory_config)

        wm.add("low relevance", relevance=0.3)
        wm.add("high relevance", relevance=0.9)

        focused = wm.get_focused()

        assert focused == "high relevance"

    def test_update_attention(self, memory_config):
        """Test updating attention weights."""
        wm = WorkingMemory(memory_config)

        wm.add("item 1")
        wm.add("item 2")
        wm.add("item 3")

        attention_scores = [0.8, 0.5, 0.3]
        wm.update_attention(attention_scores)

        assert wm.buffer[0].attention_weight == 0.8
        assert wm.buffer[1].attention_weight == 0.5
        assert wm.buffer[2].attention_weight == 0.3

    def test_rehearsal_decay(self, memory_config):
        """Test that items decay without rehearsal."""
        wm = WorkingMemory(memory_config)

        wm.add("decaying content", relevance=0.5)

        initial_activation = wm.buffer[0].activation_level

        # Wait and manually trigger rehearsal
        time.sleep(1)
        wm.rehearse()

        # Activation should have decayed
        assert wm.buffer[0].activation_level < initial_activation

    def test_phonological_loop(self, memory_config):
        """Test phonological loop for verbal information."""
        wm = WorkingMemory(memory_config)

        wm.add_to_phonological_loop("verbal information")

        assert len(wm.phonological_loop) == 1
        assert wm.phonological_loop[0]["content"] == "verbal information"

    def test_visuospatial_sketchpad(self, memory_config):
        """Test visuospatial sketchpad."""
        wm = WorkingMemory(memory_config)

        visual_data = {"type": "image", "data": "pixel_array"}
        wm.add_to_visuospatial_sketchpad(visual_data)

        assert len(wm.visuospatial_sketchpad) == 1
        assert wm.visuospatial_sketchpad[0]["content"] == visual_data

    def test_execute_task(self, memory_config):
        """Test executing task with central executive."""
        wm = WorkingMemory(memory_config)

        wm.add("task data")

        def test_task(context):
            return {"result": "success", "buffer_size": len(context["buffer"])}

        result = wm.execute_task(test_task)

        assert result is not None
        assert result["result"] == "success"

    def test_store_working_memory(self, memory_config):
        """Test storing in working memory."""
        wm = WorkingMemory(memory_config)

        memory = wm.store("test content", relevance=0.7)

        assert isinstance(memory, Memory)
        assert memory.type == MemoryType.WORKING
        assert len(wm.buffer) == 1

    def test_retrieve_working_memory(self, memory_config):
        """Test retrieving from working memory."""
        wm = WorkingMemory(memory_config)

        wm.add("retrievable content")

        # FIXED: Use keyword arguments
        query = MemoryQuery(
            query_type="working",
            content="retrievable content",
            embedding=None,
            time_range=None,
            filters={},
            limit=5,
        )

        result = wm.retrieve(query)

        assert isinstance(result, RetrievalResult)
        assert len(result.memories) >= 1

    def test_clear_working_memory(self, memory_config):
        """Test clearing working memory."""
        wm = WorkingMemory(memory_config)

        wm.add("item 1")
        wm.add("item 2")
        wm.add("item 3")

        assert len(wm.buffer) == 3

        wm.clear()

        assert len(wm.buffer) == 0
        assert wm.focus is None

    def test_consolidate_working_memory(self, memory_config):
        """Test consolidating working memory."""
        wm = WorkingMemory(memory_config)

        wm.add("important item", relevance=0.9)
        wm.add("less important", relevance=0.3)

        consolidated = wm.consolidate()

        # Should consolidate high-relevance items
        assert consolidated >= 0


# ============================================================
# INTEGRATION TESTS
# ============================================================


class TestIntegration:
    """Integration tests across memory types."""

    def test_episodic_to_semantic(self, memory_config):
        """Test converting episodic to semantic memory."""
        em = EpisodicMemory(memory_config)
        sm = SemanticMemory(memory_config)

        # Create episode with learnable concept
        em.start_episode({"learning": "new_concept"})
        em.add_event({"type": "learned", "concept": "neural_network"})
        em.end_episode(outcome="success")

        # Extract concept
        sm.add_concept(
            name="neural_network", definition="Learned from episode", confidence=0.6
        )

        concept = sm.query_concept("neural_network")

        assert concept is not None
        assert concept.name == "neural_network"

    def test_procedural_with_working_memory(self, memory_config):
        """Test procedural memory using working memory."""
        pm = ProceduralMemory(memory_config)
        wm = WorkingMemory(memory_config)

        # Add task to working memory
        wm.add({"task": "process_data"})

        # Define skill that uses working memory
        def use_working_memory(context):
            context["wm_accessed"] = True
            return context

        pm.add_skill("wm_skill", [use_working_memory])

        result = pm.execute_skill("wm_skill", {})

        assert result["wm_accessed"] is True

    def test_all_memory_types_store_retrieve(self, memory_config):
        """Test storing and retrieving across all memory types."""
        em = EpisodicMemory(memory_config)
        sm = SemanticMemory(memory_config)
        pm = ProceduralMemory(memory_config)
        wm = WorkingMemory(memory_config)

        # Store in each
        em_memory = em.store("episodic content")
        sm_memory = sm.store("semantic content")
        pm_memory = pm.store("procedural content")
        wm_memory = wm.store("working content")

        assert em_memory.type == MemoryType.EPISODIC
        assert sm_memory.type == MemoryType.SEMANTIC
        assert pm_memory.type == MemoryType.PROCEDURAL
        assert wm_memory.type == MemoryType.WORKING

        # Retrieve from each - FIXED: Use keyword arguments
        em_result = em.retrieve(
            MemoryQuery(
                query_type="episodic",
                content=None,
                embedding=None,
                time_range=None,
                filters={},
                limit=5,
            )
        )
        sm_result = sm.retrieve(
            MemoryQuery(
                query_type="semantic",
                content=None,
                embedding=None,
                time_range=None,
                filters={},
                limit=5,
            )
        )
        pm_result = pm.retrieve(
            MemoryQuery(
                query_type="procedural",
                content=None,
                embedding=None,
                time_range=None,
                filters={},
                limit=5,
            )
        )
        wm_result = wm.retrieve(
            MemoryQuery(
                query_type="working",
                content=None,
                embedding=None,
                time_range=None,
                filters={},
                limit=5,
            )
        )

        assert len(em_result.memories) >= 0
        assert len(sm_result.memories) >= 0
        assert len(pm_result.memories) >= 0
        assert len(wm_result.memories) >= 0


# ============================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_episode(self, memory_config):
        """Test ending episode with no events."""
        em = EpisodicMemory(memory_config)

        em.start_episode({})
        em.end_episode()

        # Should handle gracefully
        assert len(em.episodes) == 1

    def test_skill_execution_failure(self, memory_config):
        """Test skill execution with failing step."""
        pm = ProceduralMemory(memory_config)

        def failing_step(context):
            raise ValueError("Step failed")

        pm.add_skill("failing_skill", [failing_step])

        result = pm.execute_skill("failing_skill", {})

        # Should handle failure gracefully
        assert result is None

    def test_working_memory_overflow(self, memory_config):
        """Test working memory with many items."""
        wm = WorkingMemory(memory_config)

        # Add more items than capacity
        for i in range(wm.capacity * 2):
            wm.add(f"item {i}")

        # Should maintain capacity limit
        assert len(wm.buffer) <= wm.capacity

    def test_concept_with_no_attributes(self, memory_config):
        """Test creating concept with minimal information."""
        sm = SemanticMemory(memory_config)

        concept_id = sm.add_concept("Minimal", "")

        assert concept_id is not None
        assert concept_id in sm.concepts

    def test_retrieve_with_invalid_query(self, memory_config):
        """Test retrieval with invalid query."""
        em = EpisodicMemory(memory_config)

        # FIXED: Use keyword arguments
        query = MemoryQuery(
            query_type="invalid",
            content=None,
            embedding=None,
            time_range=None,
            filters={},
            limit=5,
        )

        result = em.retrieve(query)

        # Should handle gracefully
        assert isinstance(result, RetrievalResult)

    def test_forget_nonexistent_memory(self, memory_config):
        """Test forgetting non-existent memory."""
        em = EpisodicMemory(memory_config)

        success = em.forget("nonexistent_id")

        assert success is False

    def test_concurrent_episode_operations(self, memory_config):
        """Test concurrent operations on episodes."""
        import threading

        em = EpisodicMemory(memory_config)
        errors = []
        lock = threading.Lock()

        def add_episodes(thread_id):
            try:
                for i in range(5):
                    # Use lock to prevent race condition on start/end
                    with lock:
                        em.start_episode({"thread": f"worker_{thread_id}", "id": i})
                        em.add_event({"type": "test", "thread": thread_id})
                        time.sleep(0.001)  # Small delay to allow interleaving
                        em.end_episode()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_episodes, args=(i,)) for i in range(3)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # Should complete without errors when properly synchronized
        assert len(errors) == 0


# ============================================================
# RUN TESTS
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
