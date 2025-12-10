"""
test_knowledge_storage.py - Comprehensive tests for knowledge_storage module
Part of the VULCAN-AGI system
"""

import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

# Import the module under test
from vulcan.knowledge_crystallizer.knowledge_storage import (
    IndexEntry, KnowledgeIndex, KnowledgePruner, PrincipleVersion,
    SimpleVectorIndex, StorageBackend, VersionedKnowledgeBase)


# Mock Principle class for testing
@dataclass
class MockPrinciple:
    """Mock principle for testing"""

    id: str
    domain: str = "general"
    description: str = "Test principle"
    type: str = "test"
    confidence: float = 0.8
    success_count: int = 0
    failure_count: int = 0
    patterns: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    last_updated: float = field(default_factory=time.time)

    def to_dict(self):
        return {
            "id": self.id,
            "domain": self.domain,
            "description": self.description,
            "type": self.type,
            "confidence": self.confidence,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
        }


class TestPrincipleVersion:
    """Tests for PrincipleVersion"""

    def test_create_version(self):
        """Test creating a version"""
        principle = MockPrinciple(id="test_1", domain="math")
        version = PrincipleVersion(
            version=1,
            principle=principle,
            changes=["Added field X"],
            author="test_user",
        )

        assert version.version == 1
        assert version.principle == principle
        assert version.author == "test_user"
        assert len(version.changes) == 1

    def test_version_to_dict(self):
        """Test converting version to dict"""
        version = PrincipleVersion(
            version=1,
            changes=["Initial version"],
            author="test_user",
            commit_message="First commit",
        )

        data = version.to_dict()
        assert data["version"] == 1
        assert data["author"] == "test_user"
        assert data["commit_message"] == "First commit"

    def test_version_size(self):
        """Test getting version size"""
        principle = MockPrinciple(id="test_1")
        version = PrincipleVersion(version=1, principle=principle)

        size = version.get_size()
        assert size > 0


class TestIndexEntry:
    """Tests for IndexEntry"""

    def test_create_entry(self):
        """Test creating an index entry"""
        entry = IndexEntry(
            principle_id="test_1", domain="math", patterns=["addition", "subtraction"]
        )

        assert entry.principle_id == "test_1"
        assert entry.domain == "math"
        assert len(entry.patterns) == 2
        assert entry.access_count == 0

    def test_update_access(self):
        """Test updating access statistics"""
        entry = IndexEntry(principle_id="test_1", domain="math", patterns=[])

        initial_count = entry.access_count
        initial_time = entry.timestamp

        time.sleep(0.01)
        entry.update_access()

        assert entry.access_count == initial_count + 1
        assert entry.timestamp > initial_time


class TestSimpleVectorIndex:
    """Tests for SimpleVectorIndex"""

    def test_create_index(self):
        """Test creating a vector index"""
        index = SimpleVectorIndex(dim=128)
        assert index.dim == 128
        assert len(index.vectors) == 0

    def test_add_vectors(self):
        """Test adding vectors to index"""
        index = SimpleVectorIndex(dim=128)

        vectors = np.random.randn(5, 128).astype("float32")
        ids = ["p1", "p2", "p3", "p4", "p5"]

        index.add(vectors, ids)

        assert len(index.vectors) == 5
        assert len(index.id_to_index) == 5

    def test_search_vectors(self):
        """Test searching for similar vectors"""
        index = SimpleVectorIndex(dim=128)

        # Add some vectors
        vectors = np.random.randn(10, 128).astype("float32")
        ids = [f"p{i}" for i in range(10)]
        index.add(vectors, ids)

        # Search
        query = vectors[0]  # Should match first vector
        distances, indices = index.search(query, k=3)

        assert len(distances[0]) == 3
        assert len(indices[0]) == 3
        assert indices[0][0] == 0  # First result should be exact match

    def test_remove_vector(self):
        """Test removing a vector"""
        index = SimpleVectorIndex(dim=128)

        vectors = np.random.randn(5, 128).astype("float32")
        ids = ["p1", "p2", "p3", "p4", "p5"]
        index.add(vectors, ids)

        index.remove("p3")

        assert len(index.vectors) == 4
        assert "p3" not in index.id_to_index

    def test_clear_index(self):
        """Test clearing the index"""
        index = SimpleVectorIndex(dim=128)

        vectors = np.random.randn(5, 128).astype("float32")
        ids = ["p1", "p2", "p3", "p4", "p5"]
        index.add(vectors, ids)

        index.clear()

        assert len(index.vectors) == 0
        assert len(index.id_to_index) == 0


class TestVersionedKnowledgeBase:
    """Tests for VersionedKnowledgeBase"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    def kb(self, temp_dir):
        """Create a knowledge base for testing"""
        return VersionedKnowledgeBase(
            backend=StorageBackend.MEMORY, storage_path=temp_dir, max_versions=10
        )

    def test_create_knowledge_base(self, kb):
        """Test creating a knowledge base"""
        assert kb is not None
        assert kb.backend == StorageBackend.MEMORY
        assert kb.max_versions == 10

    def test_store_principle(self, kb):
        """Test storing a principle"""
        principle = MockPrinciple(id="test_1", domain="math")

        pid = kb.store(principle, author="test_user", message="Initial version")

        assert pid == "test_1"
        assert "test_1" in kb.principles
        assert len(kb.versions["test_1"]) == 1

    def test_store_invalid_principle(self, kb):
        """Test storing invalid principles"""
        # None principle
        with pytest.raises(ValueError):
            kb.store(None)

        # Principle with no ID
        principle = MockPrinciple(id="")
        with pytest.raises(ValueError):
            kb.store(principle)

    def test_update_principle(self, kb):
        """Test updating a principle"""
        principle = MockPrinciple(id="test_1", domain="math", confidence=0.7)
        kb.store(principle, author="user1", message="Initial")

        # Update
        principle.confidence = 0.9
        principle.description = "Updated description"
        kb.store_versioned(principle, author="user2", message="Updated confidence")

        assert kb.version_counters["test_1"] == 1
        assert len(kb.versions["test_1"]) == 2

    def test_version_pruning(self, kb):
        """Test version pruning when max_versions exceeded"""
        principle = MockPrinciple(id="test_1", domain="math")
        kb.store(principle)

        # Create many versions
        for i in range(15):
            principle.confidence += 0.01
            kb.store_versioned(principle, message=f"Version {i + 2}")

        # Should have pruned to max_versions
        assert len(kb.versions["test_1"]) <= kb.max_versions

    def test_get_principle(self, kb):
        """Test retrieving a principle"""
        principle = MockPrinciple(id="test_1", domain="math")
        kb.store(principle)

        retrieved = kb.get("test_1")

        assert retrieved is not None
        assert retrieved.id == "test_1"
        assert retrieved.domain == "math"

    def test_get_specific_version(self, kb):
        """Test retrieving a specific version"""
        principle = MockPrinciple(id="test_1", confidence=0.5)
        kb.store(principle)

        principle.confidence = 0.8
        kb.store_versioned(principle)

        # Get version 0
        v0 = kb.get("test_1", version=0)
        assert v0.confidence == 0.5

        # Get current
        current = kb.get("test_1")
        assert current.confidence == 0.8

    def test_batch_store(self, kb):
        """Test batch storing principles"""
        principles = [MockPrinciple(id=f"test_{i}", domain="math") for i in range(5)]

        ids = kb.batch_store(principles, author="batch_user")

        assert len(ids) == 5
        assert all(f"test_{i}" in kb.principles for i in range(5))

    def test_batch_get(self, kb):
        """Test batch retrieval"""
        principles = [MockPrinciple(id=f"test_{i}", domain="math") for i in range(5)]
        kb.batch_store(principles)

        results = kb.get_batch([f"test_{i}" for i in range(5)]

        assert len(results) == 5
        assert all(f"test_{i}" in results for i in range(5))

    def test_search_by_domain(self, kb):
        """Test searching by domain"""
        kb.store(MockPrinciple(id="p1", domain="math"))
        kb.store(MockPrinciple(id="p2", domain="math"))
        kb.store(MockPrinciple(id="p3", domain="physics"))

        result = kb.search({"domain": "math"}, limit=10)

        assert result.total_count == 2
        assert len(result.principles) == 2

    def test_search_by_confidence(self, kb):
        """Test searching by confidence threshold"""
        kb.store(MockPrinciple(id="p1", confidence=0.9))
        kb.store(MockPrinciple(id="p2", confidence=0.5))
        kb.store(MockPrinciple(id="p3", confidence=0.3))

        result = kb.search({"min_confidence": 0.6}, limit=10)

        assert result.total_count == 1
        assert len(result.principles) == 1

    def test_search_by_text(self, kb):
        """Test text search"""
        kb.store(MockPrinciple(id="p1", description="addition algorithm"))
        kb.store(MockPrinciple(id="p2", description="subtraction method"))
        kb.store(MockPrinciple(id="p3", description="addition principle"))

        result = kb.search({"text": "addition"}, limit=10)

        assert len(result.principles) == 2

    def test_rollback(self, kb):
        """Test rolling back to previous version"""
        principle = MockPrinciple(id="test_1", confidence=0.5)
        kb.store(principle)

        principle.confidence = 0.8
        kb.store_versioned(principle)

        principle.confidence = 0.9
        kb.store_versioned(principle)

        # Rollback to version 0
        success = kb.rollback("test_1", 0)

        assert success
        current = kb.get("test_1")
        assert current.confidence == 0.5

    def test_get_history(self, kb):
        """Test getting version history"""
        principle = MockPrinciple(id="test_1")
        kb.store(principle)

        for i in range(3):
            principle.confidence += 0.1
            kb.store_versioned(principle)

        history = kb.get_history("test_1")

        assert len(history) == 4  # Initial + 3 updates

    def test_delete_principle(self, kb):
        """Test deleting a principle"""
        principle = MockPrinciple(id="test_1")
        kb.store(principle)

        success = kb.delete("test_1", soft=True)

        assert success
        assert "test_1" not in kb.principles

    def test_find_similar(self, kb):
        """Test finding similar principles"""
        kb.store(MockPrinciple(id="p1", domain="math", type="arithmetic"))
        kb.store(MockPrinciple(id="p2", domain="math", type="arithmetic"))
        kb.store(MockPrinciple(id="p3", domain="physics", type="mechanics"))

        reference = kb.get("p1")
        similar = kb.find_similar(reference, threshold=0.3)

        assert len(similar) > 0
        assert all(p.id != "p1" for p in similar)

    def test_export_import_json(self, kb, temp_dir):
        """Test exporting and importing as JSON"""
        kb.store(MockPrinciple(id="p1", domain="math"))
        kb.store(MockPrinciple(id="p2", domain="physics"))

        export_path = temp_dir / "export.json"
        success = kb.export(export_path, format="json")

        assert success
        assert export_path.exists()

        # Create new KB and import
        kb2 = VersionedKnowledgeBase(
            backend=StorageBackend.MEMORY, storage_path=temp_dir / "kb2"
        )
        success = kb2.import_from(export_path)

        assert success
        assert len(kb2.principles) > 0

    def test_export_import_pickle(self, kb, temp_dir):
        """Test exporting and importing as pickle"""
        kb.store(MockPrinciple(id="p1", domain="math"))

        export_path = temp_dir / "export.pkl"
        success = kb.export(export_path, format="pickle")

        assert success
        assert export_path.exists()

    def test_get_statistics(self, kb):
        """Test getting storage statistics"""
        for i in range(5):
            kb.store(MockPrinciple(id=f"p{i}", domain="math"))

        stats = kb.get_statistics()

        assert stats["total_principles"] == 5
        assert stats["total_versions"] >= 5
        assert "storage_backend" in stats


class TestKnowledgeIndex:
    """Tests for KnowledgeIndex"""

    @pytest.fixture
    def index(self):
        """Create an index for testing"""
        return KnowledgeIndex(embedding_dim=128)

    def test_create_index(self, index):
        """Test creating an index"""
        assert index is not None
        assert index.embedding_dim == 128

    def test_index_by_domain(self, index):
        """Test indexing by domain"""
        principle = MockPrinciple(id="p1", domain="math")
        index.index_by_domain(principle)

        assert "p1" in index.domain_index["math"]
        assert "p1" in index.entries

    def test_index_by_pattern(self, index):
        """Test indexing by pattern"""
        principle = MockPrinciple(id="p1", patterns=["addition", "subtraction"])
        index.index_by_pattern(principle)

        assert "p1" in index.pattern_index["addition"]
        assert "p1" in index.pattern_index["subtraction"]

    def test_index_by_type(self, index):
        """Test indexing by type"""
        principle = MockPrinciple(id="p1", type="arithmetic")
        index.index_by_type(principle)

        assert "p1" in index.type_index["arithmetic"]

    def test_full_indexing(self, index):
        """Test full principle indexing"""
        principle = MockPrinciple(
            id="p1", domain="math", type="arithmetic", patterns=["addition"]
        )
        index.index_principle(principle)

        assert "p1" in index.domain_index["math"]
        assert "p1" in index.type_index["arithmetic"]
        assert "p1" in index.pattern_index["addition"]

    def test_find_relevant_by_domain(self, index):
        """Test finding relevant principles by domain"""
        index.index_principle(MockPrinciple(id="p1", domain="math"))
        index.index_principle(MockPrinciple(id="p2", domain="math"))
        index.index_principle(MockPrinciple(id="p3", domain="physics"))

        results = index.find_relevant({"domain": "math"})

        assert len(results) == 2
        assert "p1" in results
        assert "p2" in results

    def test_find_relevant_by_patterns(self, index):
        """Test finding relevant principles by patterns"""
        index.index_principle(
            MockPrinciple(id="p1", patterns=["addition", "subtraction"])
        )
        index.index_principle(MockPrinciple(id="p2", patterns=["multiplication"]))

        results = index.find_relevant({"patterns": ["addition"]})

        assert "p1" in results

    def test_search_by_similarity(self, index):
        """Test similarity search"""
        # Index some principles
        for i in range(5):
            principle = MockPrinciple(
                id=f"p{i}", domain="math", description=f"Test principle {i}"
            )
            index.index_principle(principle)

        # Search by pattern
        pattern = MockPrinciple(id="query", domain="math")
        results = index.search_by_similarity(pattern, top_k=3)

        assert len(results) <= 3
        assert all(isinstance(r, tuple) for r in results)
        assert all(len(r) == 2 for r in results)

    def test_remove_from_index(self, index):
        """Test removing from index"""
        principle = MockPrinciple(id="p1", domain="math")
        index.index_principle(principle)

        index.remove_from_index("p1")

        assert "p1" not in index.domain_index["math"]
        assert "p1" not in index.entries

    def test_get_statistics(self, index):
        """Test getting index statistics"""
        for i in range(5):
            index.index_principle(MockPrinciple(id=f"p{i}", domain="math"))

        stats = index.get_statistics()

        assert stats["total_indexed"] >= 5
        assert stats["entries"] == 5


class TestKnowledgePruner:
    """Tests for KnowledgePruner"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    def pruner(self, temp_dir):
        """Create a pruner for testing"""
        return KnowledgePruner(archive_path=temp_dir / "archive")

    @pytest.fixture
    def kb(self, temp_dir):
        """Create a knowledge base for testing"""
        return VersionedKnowledgeBase(
            backend=StorageBackend.MEMORY, storage_path=temp_dir / "kb"
        )

    def test_create_pruner(self, pruner):
        """Test creating a pruner"""
        assert pruner is not None
        assert pruner.pruned_count == 0

    def test_identify_outdated(self, pruner, kb):
        """Test identifying outdated principles"""
        # Create old principle
        principle = MockPrinciple(id="old_p1")
        kb.store(principle)

        # Fake old update time
        kb.update_times["old_p1"] = time.time() - (100 * 86400)

        candidates = pruner.identify_outdated(kb, age_threshold_days=90)

        assert len(candidates) == 1
        assert candidates[0].principle_id == "old_p1"
        assert candidates[0].reason == "outdated"

    def test_identify_low_confidence(self, pruner):
        """Test identifying low-confidence principles"""
        principles = [
            MockPrinciple(id="p1", confidence=0.9),
            MockPrinciple(id="p2", confidence=0.2),
            MockPrinciple(id="p3", confidence=0.1),
        ]

        candidates = pruner.identify_low_confidence(
            principles, confidence_threshold=0.3
        )

        assert len(candidates) == 2
        assert all(c.reason == "low_confidence" for c in candidates)

    def test_identify_contradictory(self, pruner):
        """Test identifying contradictory principles"""
        principles = [
            MockPrinciple(
                id="p1",
                domain="math",
                type="rule",
                description="Always increase value",
                confidence=0.8,
            ),
            MockPrinciple(
                id="p2",
                domain="math",
                type="rule",
                description="Never increase value",
                confidence=0.6,
            ),
            MockPrinciple(
                id="p3", domain="physics", type="law", description="Something else"
            ),
        ]

        candidates = pruner.prune_contradictory(principles)

        # Should identify the contradiction between p1 and p2
        assert len(candidates) > 0

    def test_archive_pruned(self, pruner):
        """Test archiving pruned principles"""
        principle = MockPrinciple(id="p1", domain="math")

        pruner.archive_pruned(principle)

        assert pruner.pruned_count == 1
        assert len(pruner.archive) == 1

    def test_execute_pruning(self, pruner, kb):
        """Test executing pruning"""
        # Create principles
        kb.store(MockPrinciple(id="p1", confidence=0.1))
        kb.store(MockPrinciple(id="p2", confidence=0.9))

        # Identify low confidence
        candidates = pruner.identify_low_confidence(
            list(kb.principles.values()), confidence_threshold=0.5
        )

        # Execute pruning
        pruned = pruner.execute_pruning(candidates, kb, threshold=0.5)

        assert pruned > 0
        assert "p1" not in kb.principles
        assert "p2" in kb.principles

    def test_restore_from_archive(self, pruner, kb):
        """Test restoring from archive"""
        principle = MockPrinciple(id="p1", domain="math")
        kb.store(principle)

        # Archive and delete
        pruner.archive_pruned(principle)
        kb.delete("p1")

        # Restore
        success = pruner.restore_from_archive("p1", kb)

        assert success
        # Note: Restoration behavior depends on implementation

    def test_get_statistics(self, pruner):
        """Test getting pruner statistics"""
        principle = MockPrinciple(id="p1")
        pruner.archive_pruned(principle)

        stats = pruner.get_statistics()

        assert stats["total_pruned"] == 1
        assert stats["archived_count"] == 1


class TestIntegration:
    """Integration tests combining multiple components"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)

    def test_full_workflow(self, temp_dir):
        """Test complete workflow: store, index, search, prune"""
        # Create components
        kb = VersionedKnowledgeBase(
            backend=StorageBackend.MEMORY, storage_path=temp_dir / "kb"
        )
        index = KnowledgeIndex()
        pruner = KnowledgePruner(archive_path=temp_dir / "archive")

        # Store principles
        principles = [
            MockPrinciple(
                id=f"p{i}",
                domain="math",
                confidence=0.5 + (i * 0.1),
                patterns=["arithmetic"],
            )
            for i in range(5):
        ]

        for p in principles:
            kb.store(p)
            index.index_principle(p)

        # Search
        results = index.find_relevant({"domain": "math"})
        assert len(results) == 5

        # Prune low confidence
        candidates = pruner.identify_low_confidence(
            list(kb.principles.values()), confidence_threshold=0.7
        )
        pruned = pruner.execute_pruning(candidates, kb, threshold=0.5)

        assert pruned > 0
        assert kb.total_principles < 5

    def test_versioning_workflow(self, temp_dir):
        """Test version control workflow"""
        kb = VersionedKnowledgeBase(
            backend=StorageBackend.MEMORY, storage_path=temp_dir
        )

        # Create and evolve principle
        principle = MockPrinciple(id="evolving", confidence=0.5)
        kb.store(principle, author="user1", message="Initial")

        # Update multiple times
        for i in range(3):
            principle.confidence += 0.1
            principle.success_count += 1
            kb.store_versioned(
                principle, author=f"user{i + 2}", message=f"Iteration {i + 1}"
            )

        # Verify history
        history = kb.get_history("evolving")
        assert len(history) == 4

        # Rollback
        kb.rollback("evolving", 0)
        current = kb.get("evolving")
        assert current.confidence == 0.5

    def test_persistence_workflow(self, temp_dir):
        """Test save and load workflow"""
        # Create and populate KB
        kb1 = VersionedKnowledgeBase(
            backend=StorageBackend.FILE, storage_path=temp_dir / "kb1"
        )

        for i in range(3):
            kb1.store(MockPrinciple(id=f"p{i}", domain="math"))

        kb1.save()

        # Load into new KB
        kb2 = VersionedKnowledgeBase(
            backend=StorageBackend.FILE, storage_path=temp_dir / "kb1"
        )
        kb2.load()

        assert len(kb2.principles) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
