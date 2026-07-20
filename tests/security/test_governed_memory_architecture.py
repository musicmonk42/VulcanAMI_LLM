"""Architecture gates for the canonical governed-memory closure."""
from pathlib import Path
import pytest
from vulcan.memory.governed import MemoryRuntimeConfig, SQLiteMemoryRepository
from vulcan.runtime.audit import CanonicalAudit

ROOT = Path(__file__).resolve().parents[2]

def test_runtime_does_not_import_legacy_memory_writers():
    sources = (ROOT / "src/vulcan/runtime").glob("*.py")
    forbidden = ("memory.persistence", "memory.retrieval", "memory.specialized", "memory.consolidation", "memory_bridge", "persistant_memory")
    text = "\n".join(path.read_text() for path in sources)
    assert not any(token in text for token in forbidden)

def test_second_repository_cannot_own_the_same_sqlite_store(tmp_path):
    audit = CanonicalAudit(tmp_path / "audit" / "events.jsonl")
    first = SQLiteMemoryRepository(str(tmp_path / "preferences.sqlite"), audit=audit)
    with pytest.raises(RuntimeError, match="already owned"):
        SQLiteMemoryRepository(str(tmp_path / "preferences.sqlite"), audit=audit)
    first.close(); audit.close()

def test_governed_config_rejects_unsafe_or_multi_replica_topology(tmp_path):
    with pytest.raises(RuntimeError):
        MemoryRuntimeConfig(True, tmp_path / "p.sqlite", tmp_path, replicas=2).validated()
    with pytest.raises(RuntimeError):
        MemoryRuntimeConfig(True, Path("relative.sqlite"), tmp_path).validated()
