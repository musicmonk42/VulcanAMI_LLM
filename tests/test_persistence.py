"""
Comprehensive test suite for persistence.py
"""

import json
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.persistence import (DEFAULT_CACHE_SIZE, DEFAULT_CACHE_TTL,
                             DEFAULT_MAX_CONNECTIONS, MAX_BACKUP_COUNT,
                             CacheEntry, ConnectionPool, IntegrityError,
                             KeyManagementError, KeyManager, PersistenceError,
                             PersistenceLayer, WorkingMemory)


@pytest.fixture
def temp_db_dir():
    """Create temporary database directory."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def persistence(temp_db_dir):
    """Create PersistenceLayer instance."""
    db_path = f"{temp_db_dir}/test.db"
    layer = PersistenceLayer(db_path=db_path)
    yield layer
    layer.shutdown()


@pytest.fixture
def sample_graph():
    """Create sample graph."""
    return {
        "id": "test_graph_1",
        "type": "Graph",
        "nodes": [
            {"id": "n1", "type": "Input"},
            {"id": "n2", "type": "Process"},
            {"id": "n3", "type": "Output"}
        ],
        "edges": [
            {"from": "n1", "to": "n2"},
            {"from": "n2", "to": "n3"}
        ]
    }


class TestCacheEntry:
    """Test CacheEntry class."""

    def test_cache_entry_creation(self):
        """Test creating cache entry."""
        entry = CacheEntry("test_value", 10)

        assert entry.value == "test_value"
        assert entry.expires_at > time.time()

    def test_cache_entry_expiration(self):
        """Test cache entry expiration."""
        entry = CacheEntry("test", 0)

        time.sleep(0.1)

        assert entry.is_expired() is True

    def test_cache_entry_not_expired(self):
        """Test cache entry not expired."""
        entry = CacheEntry("test", 10)

        assert entry.is_expired() is False


class TestWorkingMemory:
    """Test WorkingMemory class."""

    def test_initialization(self):
        """Test working memory initialization."""
        memory = WorkingMemory(max_size=100, default_ttl=60)

        assert memory.max_size == 100
        assert memory.default_ttl == 60
        assert len(memory.cache) == 0

    def test_store_and_recall(self):
        """Test storing and recalling values."""
        memory = WorkingMemory()

        memory.store("key1", "value1")
        result = memory.recall("key1")

        assert result == "value1"
        assert memory.hits == 1

    def test_recall_nonexistent(self):
        """Test recalling non-existent key."""
        memory = WorkingMemory()

        result = memory.recall("nonexistent")

        assert result is None
        assert memory.misses == 1

    def test_recall_expired(self):
        """Test recalling expired entry."""
        memory = WorkingMemory()

        memory.store("key1", "value1", ttl=0)
        time.sleep(0.1)

        result = memory.recall("key1")

        assert result is None
        assert memory.misses == 1

    def test_lru_eviction(self):
        """Test LRU eviction."""
        memory = WorkingMemory(max_size=2)

        memory.store("key1", "value1")
        memory.store("key2", "value2")
        memory.store("key3", "value3")  # Should evict key1

        assert memory.recall("key1") is None
        assert memory.recall("key2") == "value2"
        assert memory.recall("key3") == "value3"

    def test_invalidate(self):
        """Test invalidating key."""
        memory = WorkingMemory()

        memory.store("key1", "value1")
        memory.invalidate("key1")

        assert memory.recall("key1") is None

    def test_clear(self):
        """Test clearing cache."""
        memory = WorkingMemory()

        memory.store("key1", "value1")
        memory.store("key2", "value2")
        memory.clear()

        assert len(memory.cache) == 0
        assert memory.hits == 0
        assert memory.misses == 0

    def test_get_stats(self):
        """Test getting cache statistics."""
        memory = WorkingMemory()

        memory.store("key1", "value1")
        memory.recall("key1")
        memory.recall("nonexistent")

        stats = memory.get_stats()

        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5


class TestKeyManager:
    """Test KeyManager class."""

    def test_initialization_generates_keys(self, temp_db_dir):
        """Test key manager generates keys."""
        keys_dir = Path(temp_db_dir) / "keys"

        manager = KeyManager(keys_dir)

        assert manager.private_key is not None
        assert manager.public_key is not None
        assert manager.private_key_path.exists()
        assert manager.public_key_path.exists()

    def test_initialization_loads_existing_keys(self, temp_db_dir):
        """Test key manager loads existing keys."""
        keys_dir = Path(temp_db_dir) / "keys"

        # Create first manager
        manager1 = KeyManager(keys_dir)

        # Create second manager - should load same keys
        manager2 = KeyManager(keys_dir)

        # Keys should be the same
        assert manager1.private_key_path == manager2.private_key_path

    def test_sign_and_verify(self, temp_db_dir):
        """Test signing and verifying data."""
        keys_dir = Path(temp_db_dir) / "keys"
        manager = KeyManager(keys_dir)

        data = b"test data"
        signature = manager.sign_data(data)

        assert manager.verify_signature(data, signature) is True

    def test_verify_invalid_signature(self, temp_db_dir):
        """Test verifying invalid signature."""
        keys_dir = Path(temp_db_dir) / "keys"
        manager = KeyManager(keys_dir)

        data = b"test data"
        invalid_signature = "0" * 128

        assert manager.verify_signature(data, invalid_signature) is False

    def test_verify_tampered_data(self, temp_db_dir):
        """Test verifying tampered data."""
        keys_dir = Path(temp_db_dir) / "keys"
        manager = KeyManager(keys_dir)

        data = b"test data"
        signature = manager.sign_data(data)

        tampered_data = b"tampered data"

        assert manager.verify_signature(tampered_data, signature) is False


class TestConnectionPool:
    """Test ConnectionPool class."""

    def test_initialization(self, temp_db_dir):
        """Test connection pool initialization."""
        db_path = Path(temp_db_dir) / "test.db"

        pool = ConnectionPool(db_path, max_connections=3)

        assert pool.max_connections == 3
        assert len(pool.pool) == 0

    def test_get_connection(self, temp_db_dir):
        """Test getting connection."""
        db_path = Path(temp_db_dir) / "test.db"
        pool = ConnectionPool(db_path)

        conn = pool.get_connection()

        assert conn is not None
        assert isinstance(conn, sqlite3.Connection)

        pool.return_connection(conn)
        pool.close_all()

    def test_return_connection(self, temp_db_dir):
        """Test returning connection."""
        db_path = Path(temp_db_dir) / "test.db"
        pool = ConnectionPool(db_path)

        conn = pool.get_connection()
        pool.return_connection(conn)

        assert len(pool.pool) == 1

        pool.close_all()

    def test_close_all(self, temp_db_dir):
        """Test closing all connections."""
        db_path = Path(temp_db_dir) / "test.db"
        pool = ConnectionPool(db_path)

        conn1 = pool.get_connection()
        conn2 = pool.get_connection()
        pool.return_connection(conn1)
        pool.return_connection(conn2)

        pool.close_all()

        assert len(pool.pool) == 0


class TestPersistenceLayerInitialization:
    """Test PersistenceLayer initialization."""

    def test_initialization(self, temp_db_dir):
        """Test basic initialization."""
        db_path = f"{temp_db_dir}/test.db"

        layer = PersistenceLayer(db_path=db_path)

        assert layer.db_path.exists()
        assert layer.key_manager is not None
        assert layer.pool is not None

        layer.shutdown()

    def test_initialization_creates_tables(self, temp_db_dir):
        """Test that initialization creates tables."""
        db_path = f"{temp_db_dir}/test.db"
        layer = PersistenceLayer(db_path=db_path)

        with layer._get_connection() as conn:
            cursor = conn.cursor()

            # Check tables exist
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = [row[0] for row in cursor.fetchall()]

        assert "graphs" in tables
        assert "evolutions" in tables
        assert "knowledge" in tables
        assert "sessions" in tables

        layer.shutdown()


class TestGraphOperations:
    """Test graph storage and retrieval."""

    def test_store_graph(self, persistence, sample_graph):
        """Test storing a graph."""
        graph_id = persistence.store_graph(sample_graph)

        assert graph_id == "test_graph_1"

    def test_store_graph_with_agent(self, persistence, sample_graph):
        """Test storing graph with agent ID."""
        graph_id = persistence.store_graph(sample_graph, agent_id="agent1")

        assert graph_id == "test_graph_1"

    def test_store_graph_without_id(self, persistence):
        """Test storing graph without ID raises error."""
        graph = {"type": "Graph", "nodes": []}

        with pytest.raises(ValueError):
            persistence.store_graph(graph)

    def test_recall_graph(self, persistence, sample_graph):
        """Test recalling a graph."""
        persistence.store_graph(sample_graph)

        recalled = persistence.recall_graph("test_graph_1")

        assert recalled is not None
        assert recalled["id"] == "test_graph_1"
        assert len(recalled["nodes"]) == 3

    def test_recall_nonexistent_graph(self, persistence):
        """Test recalling non-existent graph."""
        result = persistence.recall_graph("nonexistent")

        assert result is None

    def test_recall_graph_uses_cache(self, persistence, sample_graph):
        """Test that recall uses cache."""
        persistence.store_graph(sample_graph)

        # First recall - cache miss
        recalled1 = persistence.recall_graph("test_graph_1", use_cache=True)

        # Second recall - cache hit
        recalled2 = persistence.recall_graph("test_graph_1", use_cache=True)

        assert recalled1 == recalled2
        assert persistence.working_memory.hits > 0


class TestEvolutionOperations:
    """Test evolution storage and retrieval."""

    def test_store_evolution(self, persistence):
        """Test storing evolution."""
        evolution = {
            "id": "evo1",
            "type": "add_node",
            "status": "approved"
        }

        evo_id = persistence.store_evolution(evolution)

        assert evo_id == "evo1"

    def test_store_evolution_without_id(self, persistence):
        """Test storing evolution without ID."""
        evolution = {"type": "add_node"}

        with pytest.raises(ValueError):
            persistence.store_evolution(evolution)

    def test_recall_evolution(self, persistence):
        """Test recalling evolution."""
        evolution = {
            "id": "evo1",
            "type": "add_node",
            "status": "approved"
        }

        persistence.store_evolution(evolution)
        recalled = persistence.recall_evolution("evo1")

        assert recalled is not None
        assert recalled["id"] == "evo1"

    def test_recall_nonexistent_evolution(self, persistence):
        """Test recalling non-existent evolution."""
        result = persistence.recall_evolution("nonexistent")

        assert result is None


class TestKnowledgeOperations:
    """Test knowledge storage and retrieval."""

    def test_store_knowledge(self, persistence):
        """Test storing knowledge."""
        knowledge = {"pattern": "A->B->C", "frequency": 10}

        kid = persistence.store_knowledge("patterns", knowledge)

        assert kid is not None

    def test_recall_knowledge(self, persistence):
        """Test recalling knowledge."""
        knowledge = {"pattern": "A->B->C", "frequency": 10}

        kid = persistence.store_knowledge("patterns", knowledge)
        recalled = persistence.recall_knowledge(kid)

        assert recalled is not None
        assert recalled["pattern"] == "A->B->C"

    def test_query_knowledge_by_category(self, persistence):
        """Test querying knowledge by category."""
        k1 = {"pattern": "A->B", "frequency": 5}
        k2 = {"pattern": "C->D", "frequency": 3}

        persistence.store_knowledge("patterns", k1)
        persistence.store_knowledge("patterns", k2)

        results = persistence.query_knowledge_by_category("patterns")

        assert len(results) == 2


class TestFeatureExtraction:
    """Test feature extraction."""

    def test_extract_features(self, persistence, sample_graph):
        """Test extracting features from graph."""
        features = persistence._extract_features(sample_graph)

        assert features["node_count"] == 3
        assert features["edge_count"] == 2
        assert "Input" in features["node_types"]

    def test_detect_cycles_no_cycle(self, persistence):
        """Test cycle detection with no cycle."""
        nodes = [
            {"id": "n1"},
            {"id": "n2"},
            {"id": "n3"}
        ]
        edges = [
            {"from": "n1", "to": "n2"},
            {"from": "n2", "to": "n3"}
        ]

        has_cycle = persistence._detect_cycles(nodes, edges)

        assert has_cycle is False

    def test_detect_cycles_with_cycle(self, persistence):
        """Test cycle detection with cycle."""
        nodes = [
            {"id": "n1"},
            {"id": "n2"},
            {"id": "n3"}
        ]
        edges = [
            {"from": "n1", "to": "n2"},
            {"from": "n2", "to": "n3"},
            {"from": "n3", "to": "n1"}
        ]

        has_cycle = persistence._detect_cycles(nodes, edges)

        assert has_cycle is True


class TestQueryGraphsByFeatures:
    """Test querying graphs by features."""

    def test_query_by_node_count(self, persistence):
        """Test querying by node count."""
        # Store graphs with different node counts
        graph1 = {
            "id": "g1",
            "nodes": [{"id": f"n{i}", "type": "Node"} for i in range(5)],
            "edges": []
        }
        graph2 = {
            "id": "g2",
            "nodes": [{"id": f"n{i}", "type": "Node"} for i in range(15)],
            "edges": []
        }

        persistence.store_graph(graph1)
        persistence.store_graph(graph2)

        results = persistence.query_graphs_by_features(node_count=10, op='>')

        assert len(results) == 1
        assert results[0]["id"] == "g2"

    def test_query_by_agent_id(self, persistence, sample_graph):
        """Test querying by agent ID."""
        persistence.store_graph(sample_graph, agent_id="agent1")

        results = persistence.query_graphs_by_features(agent_id="agent1")

        assert len(results) == 1
        assert results[0]["id"] == "test_graph_1"


class TestBackupAndRecovery:
    """Test backup and recovery."""

    def test_backup_creates_file(self, persistence):
        """Test backup creates backup file."""
        persistence.backup()

        backups = list(persistence.backup_path.glob("backup_*.db"))

        assert len(backups) > 0

    def test_backup_creates_signature(self, persistence):
        """Test backup creates signature file."""
        persistence.backup()

        backups = list(persistence.backup_path.glob("backup_*.db"))
        sig_file = backups[0].with_suffix(".sig")

        assert sig_file.exists()

    def test_backup_rotation(self, temp_db_dir):
        """Test backup rotation."""
        db_path = f"{temp_db_dir}/test.db"
        layer = PersistenceLayer(db_path=db_path)

        # Create more backups than limit
        for i in range(MAX_BACKUP_COUNT + 3):
            layer.backup()
            time.sleep(0.01)  # Ensure different timestamps

        backups = list(layer.backup_path.glob("backup_*.db"))

        # Should not exceed max count
        assert len(backups) <= MAX_BACKUP_COUNT

        layer.shutdown()

    def test_recover_from_backup(self, persistence, sample_graph):
        """Test recovering from backup."""
        # Store data and create backup
        persistence.store_graph(sample_graph)
        persistence.backup()

        # Get backup file
        backups = list(persistence.backup_path.glob("backup_*.db"))
        backup_file = str(backups[0])

        # Recover
        persistence.recover(backup_file)

        # Verify data still exists
        recalled = persistence.recall_graph("test_graph_1")
        assert recalled is not None

    def test_recover_nonexistent_backup(self, persistence):
        """Test recovering from non-existent backup."""
        with pytest.raises(FileNotFoundError):
            persistence.recover("nonexistent.db")


class TestIntegrityVerification:
    """Test integrity verification."""

    def test_verify_integrity_success(self, persistence, sample_graph):
        """Test successful integrity verification."""
        persistence.store_graph(sample_graph)

        # Should not raise
        persistence.verify_integrity()

    def test_verify_integrity_with_force(self, persistence, sample_graph):
        """Test forced integrity verification."""
        persistence.store_graph(sample_graph)

        # Force re-verification
        persistence.verify_integrity(force=True)


class TestSignatureVerification:
    """Test signature verification."""

    def test_sign_and_verify_data(self, persistence):
        """Test signing and verifying data."""
        data = b"test data"

        signature = persistence._sign_data(data)
        result = persistence._verify_signature(data, signature)

        assert result is True

    def test_verify_invalid_signature(self, persistence):
        """Test verifying invalid signature."""
        data = b"test data"
        invalid_sig = "0" * 128

        result = persistence._verify_signature(data, invalid_sig)

        assert result is False

    def test_signature_cache(self, persistence):
        """Test signature verification caching."""
        data = b"test data"
        signature = persistence._sign_data(data)

        # First verification - cache miss
        result1 = persistence._verify_signature(data, signature, use_cache=True)

        # Second verification - cache hit
        result2 = persistence._verify_signature(data, signature, use_cache=True)

        assert result1 is True
        assert result2 is True
        assert len(persistence.signature_cache) > 0


class TestAuditLogging:
    """Test audit logging."""

    def test_audit_log_creation(self, persistence):
        """Test audit log is created."""
        persistence._audit_log("test_event", {"detail": "test"})

        assert persistence.audit_log_path.exists()

    def test_audit_log_content(self, persistence):
        """Test audit log content."""
        persistence._audit_log("test_event", {"detail": "test"})

        with open(persistence.audit_log_path, encoding="utf-8") as f:
            line = f.readline()
            entry = json.loads(line)

        assert "entry" in entry
        assert "signature" in entry
        assert entry["entry"]["event"] == "test_event"


class TestStatistics:
    """Test statistics."""

    def test_get_statistics(self, persistence, sample_graph):
        """Test getting statistics."""
        persistence.store_graph(sample_graph)

        stats = persistence.get_statistics()

        assert "graphs_count" in stats
        assert "db_size_bytes" in stats
        assert "cache" in stats
        assert stats["graphs_count"] >= 1


class TestShutdown:
    """Test shutdown."""

    def test_shutdown(self, temp_db_dir):
        """Test clean shutdown."""
        db_path = f"{temp_db_dir}/test.db"
        layer = PersistenceLayer(db_path=db_path)

        # Should not raise
        layer.shutdown()


class TestExceptions:
    """Test custom exceptions."""

    def test_persistence_error(self):
        """Test PersistenceError."""
        error = PersistenceError("test error")

        assert str(error) == "test error"

    def test_integrity_error(self):
        """Test IntegrityError."""
        error = IntegrityError("integrity failed")

        assert str(error) == "integrity failed"

    def test_key_management_error(self):
        """Test KeyManagementError."""
        error = KeyManagementError("key error")

        assert str(error) == "key error"


class TestConstants:
    """Test module constants."""

    def test_constants_exist(self):
        """Test that all constants are defined."""
        assert DEFAULT_MAX_CONNECTIONS > 0
        assert DEFAULT_CACHE_SIZE > 0
        assert DEFAULT_CACHE_TTL > 0
        assert MAX_BACKUP_COUNT > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
