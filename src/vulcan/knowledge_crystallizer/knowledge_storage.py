"""
knowledge_storage.py - Knowledge storage and management for Knowledge Crystallizer
Part of the VULCAN-AGI system
"""

import copy
import difflib
import gzip
import hashlib
import json
import logging
import pickle
import shutil
import sqlite3
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..security_fixes import safe_pickle_load

# Optional imports with fallbacks
try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("faiss not available, vector search will be limited")

logger = logging.getLogger(__name__)


class StorageBackend(Enum):
    """Storage backend types"""

    MEMORY = "memory"
    SQLITE = "sqlite"
    FILE = "file"
    HYBRID = "hybrid"  # Memory + persistent storage


class CompressionType(Enum):
    """Compression types for storage"""

    NONE = "none"
    GZIP = "gzip"
    PICKLE = "pickle"


@dataclass
class PrincipleVersion:
    """Version of a principle"""

    version: int
    principle: Any = None  # Full principle object (for backward compatibility)
    principle_diff: Optional[List[str]] = None  # Diff from base version
    base_version: Optional[int] = None  # Base version for diff
    timestamp: float = field(default_factory=time.time)
    changes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    author: Optional[str] = None  # Who made the change
    commit_message: Optional[str] = None  # Description of changes

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "version": self.version,
            "timestamp": self.timestamp,
            "changes": self.changes,
            "metadata": self.metadata,
            "author": self.author,
            "commit_message": self.commit_message,
            "has_diff": self.principle_diff is not None,
            "base_version": self.base_version,
        }

    def get_size(self) -> int:
        """Get approximate size in bytes"""
        if self.principle is not None:
            return len(pickle.dumps(self.principle))
        elif self.principle_diff is not None:
            return len(pickle.dumps(self.principle_diff))
        return 0

    def reconstruct_principle(self, base_principle: Any) -> Any:
        """Reconstruct principle from diff"""
        if self.principle is not None:
            return self.principle

        if self.principle_diff is None or base_principle is None:
            return None

        # Apply diff to reconstruct
        base_str = json.dumps(
            asdict(base_principle)
            if hasattr(base_principle, "__dataclass_fields__")
            else vars(base_principle),
            sort_keys=True,
        )

        # Apply unified diff
        base_str.splitlines()

        # Simple diff application (would need more sophisticated logic for production)
        # For now, return base if diff application fails
        try:
            # This is a simplified reconstruction - proper implementation would use difflib.patch
            return base_principle
        except Exception:
            return base_principle


@dataclass
class IndexEntry:
    """Entry in knowledge index"""

    principle_id: str
    domain: str
    patterns: List[str]
    embedding: Optional[np.ndarray] = None
    timestamp: float = field(default_factory=time.time)
    relevance_score: float = 1.0
    access_count: int = 0
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "principle_id": self.principle_id,
            "domain": self.domain,
            "patterns": self.patterns,
            "timestamp": self.timestamp,
            "relevance_score": self.relevance_score,
            "access_count": self.access_count,
            "tags": self.tags,
        }

    def update_access(self):
        """Update access statistics"""
        self.access_count += 1
        self.timestamp = time.time()


@dataclass
class PruneCandidate:
    """Candidate for pruning"""

    principle_id: str
    reason: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    recoverable: bool = True  # Can be recovered from archive


@dataclass
class QueryResult:
    """Result from a query operation"""

    principles: List[Any]
    scores: List[float]
    total_count: int
    query_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# Simple vector index implementation for when faiss is not available
class SimpleVectorIndex:
    """Simple vector index as fallback for FAISS"""

    def __init__(self, dim: int):
        self.dim = dim
        self.vectors = np.array([]).reshape(0, dim)  # Initialize properly
        self.id_to_index = {}  # Map principle_id to current index
        self.index_to_id = {}  # Map index to principle_id
        self.lock = threading.RLock()

    def add(self, vectors: np.ndarray, principle_ids: Optional[List[str]] = None):
        """Add vectors to index"""
        with self.lock:
            if len(vectors.shape) == 1:
                vectors = vectors.reshape(1, -1)

            start_idx = len(self.vectors)

            if len(self.vectors) == 0:
                self.vectors = vectors.copy()
            else:
                self.vectors = np.vstack([self.vectors, vectors])

            # Update mappings if IDs provided
            if principle_ids:
                for i, pid in enumerate(principle_ids):
                    idx = start_idx + i
                    self.id_to_index[pid] = idx
                    self.index_to_id[idx] = pid

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors"""
        with self.lock:
            if len(self.vectors) == 0:
                return np.array([[]]), np.array([[]])

            if len(query.shape) == 1:
                query = query.reshape(1, -1)

            # Calculate L2 distances
            distances = np.sum((self.vectors - query) ** 2, axis=1)

            # Get k nearest
            k = min(k, len(distances))
            indices = np.argsort(distances)[:k]

            return distances[indices].reshape(1, -1), indices.reshape(1, -1)

    def remove(self, principle_id: str):
        """Remove vector by principle ID"""
        with self.lock:
            if principle_id not in self.id_to_index:
                return

            idx = self.id_to_index[principle_id]

            # Remove vector
            self.vectors = np.delete(self.vectors, idx, axis=0)

            # Update all mappings
            del self.id_to_index[principle_id]
            del self.index_to_id[idx]

            # Shift all indices after removed index
            new_id_to_index = {}
            new_index_to_id = {}

            for old_idx, pid in self.index_to_id.items():
                new_idx = old_idx if old_idx < idx else old_idx - 1
                new_id_to_index[pid] = new_idx
                new_index_to_id[new_idx] = pid

            self.id_to_index = new_id_to_index
            self.index_to_id = new_index_to_id

    def clear(self):
        """Clear all vectors"""
        with self.lock:
            self.vectors = np.array([]).reshape(0, self.dim)
            self.id_to_index.clear()
            self.index_to_id.clear()


class VersionedKnowledgeBase:
    """Stores principles with version control"""

    def __init__(
        self,
        backend: StorageBackend = StorageBackend.MEMORY,
        storage_path: Optional[Path] = None,
        compression: CompressionType = CompressionType.NONE,
        auto_save: bool = True,
        max_versions: int = 100,
    ):
        """
        Initialize versioned knowledge base

        Args:
            backend: Storage backend type
            storage_path: Path for file/sqlite storage
            compression: Compression type for storage
            auto_save: Automatically save changes
            max_versions: Maximum versions to keep per principle
        """
        self.backend = backend
        self.storage_path = storage_path or Path("knowledge_base")
        self.compression = compression
        self.auto_save = auto_save
        self.max_versions = max_versions

        # In-memory storage
        self.principles = {}  # principle_id -> current principle
        self.versions = defaultdict(list)  # principle_id -> [PrincipleVersion]
        self.version_counters = defaultdict(int)  # principle_id -> current version

        # Metadata
        self.creation_times = {}
        self.update_times = {}
        self.access_counts = defaultdict(int)
        self.access_times = defaultdict(lambda: deque(maxlen=100))

        # Indices
        self.tag_index = defaultdict(set)  # tag -> set of principle_ids
        self.author_index = defaultdict(set)  # author -> set of principle_ids

        # Database connection pool (if using SQLite)
        self.conn_pool = []
        self.pool_lock = threading.Lock()
        self.pool_available = []
        self.pool_size = 5

        # Thread safety
        self.lock = threading.RLock()

        # Statistics
        self.total_principles = 0
        self.total_versions = 0
        self.total_storage_size = 0

        # Initialize backend
        self._initialize_backend()

        # Load existing data if available
        if self.storage_path.exists():
            self.load()

        logger.info("VersionedKnowledgeBase initialized with backend: %s", backend)

    def _get_connection(self):
        """Get connection from pool"""
        with self.pool_lock:
            if self.pool_available:
                idx = self.pool_available.pop()
                return idx, self.conn_pool[idx]
            else:
                raise RuntimeError("No available database connections in pool")

    def _release_connection(self, idx: int):
        """Release connection back to pool"""
        with self.pool_lock:
            if idx not in self.pool_available:
                self.pool_available.append(idx)

    def store(
        self, principle, author: Optional[str] = None, message: Optional[str] = None
    ) -> str:
        """
        Store principle (creates new or updates existing)

        Args:
            principle: Principle to store
            author: Author of the change
            message: Commit message

        Returns:
            Principle ID
        """
        with self.lock:
            # Validate principle
            if principle is None:
                raise ValueError("Cannot store None principle")

            principle_id = getattr(principle, "id", str(principle))

            if not principle_id or principle_id == "None":
                raise ValueError("Principle must have valid ID")

            if principle_id in self.principles:
                # Update existing
                return self.store_versioned(principle, author, message)
            else:
                # Create new - FIX: Store deep copy to avoid reference issues
                self.principles[principle_id] = copy.deepcopy(principle)
                self.creation_times[principle_id] = time.time()
                self.update_times[principle_id] = time.time()

                # Create initial version with commit data
                commit_message = message or "Initial version"
                version_data = {
                    "version": 0,
                    "principle": copy.deepcopy(principle),
                    "timestamp": time.time(),
                    "changes": ["Initial version"],
                    "author": author,
                    "commit_message": commit_message,
                }
                version = PrincipleVersion(**version_data)

                self.versions[principle_id].append(version)
                self.version_counters[principle_id] = 0

                # Update indices
                if author:
                    self.author_index[author].add(principle_id)

                # Persist if needed
                if self.auto_save:
                    self._persist_principle(principle_id, self.principles[principle_id])

                self.total_principles += 1
                self.total_versions += 1
                self.total_storage_size += version.get_size()

                logger.debug("Stored new principle: %s", principle_id)

                return principle_id

    def store_principle(self, principle) -> str:
        """
        Store principle (alias for compatibility)

        Args:
            principle: Principle to store

        Returns:
            Principle ID
        """
        return self.store(principle)

    def _compute_diff(self, old, new):
        """Compute difference between principles"""
        try:
            old_str = json.dumps(
                asdict(old) if hasattr(old, "__dataclass_fields__") else vars(old),
                sort_keys=True,
            )
            new_str = json.dumps(
                asdict(new) if hasattr(new, "__dataclass_fields__") else vars(new),
                sort_keys=True,
            )
            return list(
                difflib.unified_diff(
                    old_str.splitlines(), new_str.splitlines(), lineterm=""
                )
            )
        except Exception as e:
            logger.warning("Failed to compute diff: %s", e)
            return []

    def _prune_versions(self, principle_id: str):
        """Prune versions keeping evenly distributed samples"""
        versions = self.versions[principle_id]
        if len(versions) <= self.max_versions:
            return

        keep_indices = [0]  # Always keep first

        # Keep evenly distributed middle versions
        if self.max_versions > 2:
            step = (len(versions) - 1) / (self.max_versions - 2)
            for i in range(1, self.max_versions - 1):
                idx = int(i * step)
                if idx not in keep_indices and idx < len(versions):
                    keep_indices.append(idx)

        keep_indices.append(len(versions) - 1)  # Always keep last
        keep_indices = sorted(list(set(keep_indices)))  # Remove duplicates and sort

        # Calculate storage freed
        for i, v in enumerate(versions):
            if i not in keep_indices:
                self.total_storage_size -= v.get_size()

        self.versions[principle_id] = [versions[i] for i in keep_indices]

    def store_versioned(
        self, principle, author: Optional[str] = None, message: Optional[str] = None
    ) -> str:
        """
        Store new version of existing principle

        Args:
            principle: Updated principle
            author: Author of the change
            message: Commit message

        Returns:
            Principle ID
        """
        with self.lock:
            # Validate principle
            if principle is None:
                raise ValueError("Cannot store None principle")

            principle_id = getattr(principle, "id", str(principle))

            if not principle_id or principle_id == "None":
                raise ValueError("Principle must have valid ID")

            if principle_id not in self.principles:
                # New principle
                return self.store(principle, author, message)

            # FIX: Get the STORED version (which is already a deep copy)
            current = self.principles[principle_id]

            # FIX: Create a deep copy of the incoming principle for comparison
            # This ensures we're comparing the stored state vs the new state
            new_principle = copy.deepcopy(principle)

            # Detect changes by comparing stored version with new version
            changes = self._detect_changes(current, new_principle)

            # FIX: Allow version creation if explicitly requested
            if not changes and not message and not author:
                logger.debug("No changes detected for principle %s", principle_id)
                return principle_id

            # If changes list is empty but we have a message, add generic change
            if not changes:
                changes = ["Manual version update"]

            # Increment version
            new_version = self.version_counters[principle_id] + 1
            self.version_counters[principle_id] = new_version

            # Compute diff from current version
            diff = self._compute_diff(current, new_principle)

            # Create version entry with diff
            commit_message = message
            version_data = {
                "version": new_version,
                "principle_diff": diff if len(diff) > 0 else None,
                "base_version": new_version - 1,
                "timestamp": time.time(),
                "changes": changes,
                "author": author,
                "commit_message": commit_message,
            }

            # Store full principle for recent versions, diff for older ones
            if new_version % 10 == 0:  # Every 10th version is full
                version_data["principle"] = copy.deepcopy(new_principle)
                version_data["principle_diff"] = None
                version_data["base_version"] = None

            version = PrincipleVersion(**version_data)

            # Store version with pruning
            self.versions[principle_id].append(version)
            if len(self.versions[principle_id]) > self.max_versions:
                self._prune_versions(principle_id)

            # FIX: Update current with deep copy of new principle
            self.principles[principle_id] = copy.deepcopy(new_principle)
            self.update_times[principle_id] = time.time()

            # Update indices
            if author:
                self.author_index[author].add(principle_id)

            # Persist
            if self.auto_save:
                self._persist_principle(principle_id, self.principles[principle_id])
                self._persist_version(principle_id, version)

            self.total_versions += 1
            self.total_storage_size += version.get_size()

            logger.debug(
                "Stored version %d of principle %s with %d changes",
                new_version,
                principle_id,
                len(changes),
            )

            # Persist versioned file if using file backend
            if self.auto_save and self.backend == StorageBackend.FILE:
                version_path = self.storage_path / f"{principle_id}_v{new_version}.pkl"
                with open(version_path, "wb") as f:
                    pickle.dump(version, f)

            return principle_id

    def batch_store(
        self, principles: List[Any], author: Optional[str] = None
    ) -> List[str]:
        """
        Store multiple principles efficiently

        Args:
            principles: List of principles to store
            author: Author of the changes

        Returns:
            List of principle IDs
        """
        with self.lock:
            ids = []
            for principle in principles:
                if principle is not None:
                    try:
                        pid = self.store(principle, author, "Batch store")
                        ids.append(pid)
                    except Exception as e:
                        logger.error("Failed to store principle in batch: %s", e)

            # Single persist operation
            if self.auto_save:
                self.save()

            return ids

    def get(self, principle_id: str, version: Optional[int] = None):
        """
        Get principle by ID and optional version

        Args:
            principle_id: Principle ID
            version: Specific version (None for current)

        Returns:
            Principle or None
        """
        with self.lock:
            # Validate input
            if not principle_id or principle_id == "None":
                return None

            # Track access
            self.access_counts[principle_id] += 1
            self.access_times[principle_id].append(time.time())

            if version is None:
                # Get current version
                return self.principles.get(principle_id)
            else:
                # Get specific version
                if principle_id in self.versions:
                    for v in self.versions[principle_id]:
                        if v.version == version:
                            # Reconstruct if stored as diff
                            if v.principle is not None:
                                return v.principle
                            elif (
                                v.principle_diff is not None
                                and v.base_version is not None
                            ):
                                # Find base version and reconstruct
                                base = None
                                for base_v in self.versions[principle_id]:
                                    if base_v.version == v.base_version:
                                        base = base_v.principle
                                        break
                                if base:
                                    return v.reconstruct_principle(base)
                            return v.principle

            return None

    def get_batch(self, principle_ids: List[str]) -> Dict[str, Any]:
        """
        Get multiple principles efficiently

        Args:
            principle_ids: List of principle IDs

        Returns:
            Dictionary mapping IDs to principles
        """
        with self.lock:
            results = {}
            for pid in principle_ids:
                if pid and pid != "None":
                    principle = self.get(pid)
                    if principle:
                        results[pid] = principle
            return results

    def search(self, query: Dict[str, Any], limit: int = 10) -> QueryResult:
        """
        Search for principles

        Args:
            query: Search query with filters
            limit: Maximum results

        Returns:
            Query result
        """
        start_time = time.time()

        with self.lock:
            results = []
            scores = []

            # Validate limit
            limit = max(1, min(limit, 1000))

            # Apply filters
            candidates = list(self.principles.values())

            # Domain filter
            if "domain" in query:
                candidates = [
                    p
                    for p in candidates
                    if getattr(p, "domain", None) == query["domain"]
                ]

            # Confidence filter
            if "min_confidence" in query:
                candidates = [
                    p
                    for p in candidates
                    if getattr(p, "confidence", 0) >= query["min_confidence"]
                ]

            # Tag filter
            if "tags" in query:
                query_tags = set(query["tags"])
                tagged_ids = set()
                for tag in query_tags:
                    tagged_ids.update(self.tag_index.get(tag, set()))
                candidates = [
                    p for p in candidates if getattr(p, "id", str(p)) in tagged_ids
                ]

            # Author filter
            if "author" in query:
                author_ids = self.author_index.get(query["author"], set())
                candidates = [
                    p for p in candidates if getattr(p, "id", str(p)) in author_ids
                ]

            # Text search in description
            if "text" in query:
                search_text = query["text"].lower()
                scored_candidates = []
                for p in candidates:
                    if hasattr(p, "description"):
                        desc = str(p.description).lower()
                        if search_text in desc:
                            # Simple relevance scoring
                            score = desc.count(search_text) / max(1, len(desc.split()))
                            scored_candidates.append((p, score))

                # Sort by relevance
                scored_candidates.sort(key=lambda x: x[1], reverse=True)
                results = [p for p, _ in scored_candidates[:limit]]
                scores = [s for _, s in scored_candidates[:limit]]
            else:
                # No text search, return first N
                results = candidates[:limit]
                scores = [1.0] * len(results)

            return QueryResult(
                principles=results,
                scores=scores,
                total_count=len(candidates),
                query_time=time.time() - start_time,
                metadata={"query": query},
            )

    def rollback(self, principle_id: str, target_version: int) -> bool:
        """
        Rollback principle to target version

        Args:
            principle_id: Principle ID
            target_version: Target version number

        Returns:
            True if successful
        """
        with self.lock:
            if principle_id not in self.versions:
                logger.warning("Principle %s not found for rollback", principle_id)
                return False

            if target_version < 0:
                logger.warning("Invalid target version %d", target_version)
                return False

            # Find target version
            target = None
            for v in self.versions[principle_id]:
                if v.version == target_version:
                    target = v
                    break

            if not target:
                logger.warning(
                    "Version %d not found for principle %s",
                    target_version,
                    principle_id,
                )
                return False

            # Get target principle (reconstruct if needed)
            target_principle = None
            if target.principle is not None:
                target_principle = target.principle
            elif target.principle_diff is not None:
                # Reconstruct from diff
                base = None
                for v in self.versions[principle_id]:
                    if v.version == target.base_version:
                        base = v.principle
                        break
                if base:
                    target_principle = target.reconstruct_principle(base)

            if not target_principle:
                logger.warning("Cannot reconstruct target version %d", target_version)
                return False

            # Create rollback version
            current = self.principles[principle_id]
            rollback_changes = [
                f"Rollback from version {self.version_counters[principle_id]} to {target_version}"
            ]

            # Store current as version before rollback
            self.store_versioned(
                current, author="system", message="Pre-rollback backup"
            )

            # Rollback
            self.principles[principle_id] = copy.deepcopy(target_principle)
            self.update_times[principle_id] = time.time()

            # Create rollback version entry
            new_version = self.version_counters[principle_id] + 1
            self.version_counters[principle_id] = new_version

            version = PrincipleVersion(
                version=new_version,
                principle=copy.deepcopy(target_principle),
                timestamp=time.time(),
                changes=rollback_changes,
                metadata={
                    "rollback_from": new_version - 1,
                    "rollback_to": target_version,
                },
                author="system",
                commit_message=f"Rollback to version {target_version}",
            )

            self.versions[principle_id].append(version)

            # Persist
            if self.auto_save:
                self._persist_principle(principle_id, target_principle)
                self._persist_version(principle_id, version)

            logger.info(
                "Rolled back principle %s to version %d", principle_id, target_version
            )

            return True

    def get_history(self, principle_id: str) -> List[PrincipleVersion]:
        """
        Get version history of principle

        Args:
            principle_id: Principle ID

        Returns:
            List of versions
        """
        with self.lock:
            return self.versions.get(principle_id, []).copy()

    def get_all_principles(self) -> List[Any]:
        """Get all current principles"""
        with self.lock:
            return list(self.principles.values())

    def get_principle(self, principle_id: str):
        """Get principle by ID (alias for get)"""
        return self.get(principle_id)

    def update_principle(self, principle):
        """Update principle (alias for store_versioned)"""
        return self.store_versioned(principle)

    def delete(self, principle_id: str, soft: bool = True) -> bool:
        """
        Delete principle

        Args:
            principle_id: Principle ID to delete
            soft: If True, archive instead of permanent delete

        Returns:
            True if successful
        """
        with self.lock:
            if principle_id not in self.principles:
                return False

            if soft:
                # Archive the principle
                archived = self.principles[principle_id]
                self._archive_principle(principle_id, archived)

            # Remove from active storage
            del self.principles[principle_id]

            # Update statistics
            self.total_principles -= 1

            # Clean up indices
            for author, pids in list(self.author_index.items()):
                pids.discard(principle_id)
                if not pids:
                    del self.author_index[author]

            # Persist changes
            if self.auto_save:
                self.save()

            logger.info("Deleted principle %s (soft=%s)", principle_id, soft)
            return True

    def find_similar(self, principle, threshold: float = 0.7) -> List[Any]:
        """
        Find similar principles

        Args:
            principle: Reference principle
            threshold: Similarity threshold

        Returns:
            List of similar principles
        """
        with self.lock:
            if principle is None:
                return []

            similar = []
            principle_id = getattr(principle, "id", None)

            for pid, p in self.principles.items():
                if pid == principle_id:
                    continue

                # Calculate similarity
                similarity = self._calculate_similarity(principle, p)

                if similarity >= threshold:
                    similar.append(p)

            # Sort by similarity
            similar.sort(
                key=lambda p: self._calculate_similarity(principle, p), reverse=True
            )

            return similar

    def export(self, path: Path, format: str = "json") -> bool:
        """
        Export knowledge base

        Args:
            path: Export path
            format: Export format ('json', 'pickle')

        Returns:
            True if successful
        """
        with self.lock:
            try:
                data = {
                    "principles": {},
                    "versions": {},
                    "metadata": {
                        "total_principles": self.total_principles,
                        "total_versions": self.total_versions,
                        "export_time": time.time(),
                        "backend": self.backend.value,
                    },
                }

                # Export principles
                for pid, principle in self.principles.items():
                    if hasattr(principle, "to_dict"):
                        data["principles"][pid] = principle.to_dict()
                    else:
                        data["principles"][pid] = str(principle)

                # Export versions
                for pid, versions in self.versions.items():
                    data["versions"][pid] = [v.to_dict() for v in versions]

                # Write to file
                path.parent.mkdir(parents=True, exist_ok=True)

                if format == "json":
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
                elif format == "pickle":
                    with open(path, "wb") as f:
                        pickle.dump(data, f)

                # Compress if requested
                if self.compression == CompressionType.GZIP:
                    with open(path, "rb") as f_in:
                        with gzip.open(path.with_suffix(".gz", encoding="utf-8"), "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    path.unlink()  # Remove uncompressed file
                    logger.info(
                        "Exported knowledge base to %s (compressed)",
                        path.with_suffix(".gz"),
                    )
                else:
                    logger.info("Exported knowledge base to %s", path)

                return True

            except Exception as e:
                logger.error("Failed to export knowledge base: %s", e)
                return False

    def import_from(self, path: Path) -> bool:
        """
        Import knowledge base

        Args:
            path: Import path

        Returns:
            True if successful
        """
        with self.lock:
            try:
                # Check for compressed file
                if path.suffix == ".gz":
                    with gzip.open(path, "rb") as f:
                        content = f.read()
                    # Determine format from decompressed content
                    try:
                        data = json.loads(content)
                    except Exception:
                        data = pickle.loads(content)
                else:
                    # Load based on extension
                    if path.suffix == ".json":
                        with open(path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                    else:
                        with open(path, "rb") as f:
                            data = safe_pickle_load(f)

                # Import principles
                for pid, principle_data in data.get("principles", {}).items():
                    # Reconstruct principle (simplified - needs proper deserialization)
                    self.principles[pid] = principle_data

                # Import metadata
                metadata = data.get("metadata", {})
                self.total_principles = metadata.get(
                    "total_principles", len(self.principles)
                )
                self.total_versions = metadata.get("total_versions", 0)

                logger.info("Imported knowledge base from %s", path)
                return True

            except Exception as e:
                logger.error("Failed to import knowledge base: %s", e)
                return False

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get storage statistics

        Returns:
            Dictionary of statistics
        """
        with self.lock:
            version_counts = [len(v) for v in self.versions.values()]

            # Calculate access patterns
            recent_accesses = []
            for times in self.access_times.values():
                recent_accesses.extend(list(times))

            # Access frequency analysis
            if recent_accesses:
                recent_accesses.sort()
                time_range = recent_accesses[-1] - recent_accesses[0]
                access_rate = (
                    len(recent_accesses) / max(1, time_range) if time_range > 0 else 0
                )
            else:
                access_rate = 0

            return {
                "total_principles": self.total_principles,
                "total_versions": self.total_versions,
                "total_storage_size_bytes": self.total_storage_size,
                "avg_versions_per_principle": np.mean(version_counts)
                if version_counts
                else 0,
                "max_versions": max(version_counts) if version_counts else 0,
                "storage_backend": self.backend.value,
                "compression": self.compression.value,
                "most_accessed": self._get_most_accessed(5),
                "total_authors": len(self.author_index),
                "access_rate_per_second": access_rate,
                "cache_status": {
                    "principles_in_memory": len(self.principles),
                    "versions_in_memory": sum(len(v) for v in self.versions.values()),
                },
            }

    def save(self, path: Optional[Path] = None):
        """Save knowledge base to persistent storage"""
        save_path = path or self.storage_path

        with self.lock:
            if self.backend == StorageBackend.MEMORY:
                # Export to file for memory backend
                self.export(save_path / "knowledge_export.json")
            elif self.backend == StorageBackend.SQLITE:
                # Commit any pending changes
                for idx in range(len(self.conn_pool)):
                    try:
                        self.conn_pool[idx].commit()
                    except Exception as e:
                        logger.debug(
                            f"{self.__class__.__name__ if hasattr(self, '__class__') else 'Operation'} error: {e}"
                        )
            elif self.backend == StorageBackend.FILE:
                # Save all principles to files
                for pid, principle in self.principles.items():
                    self._persist_principle(pid, principle)

            logger.debug("Saved knowledge base")

    def load(self, path: Optional[Path] = None):
        """Load knowledge base from persistent storage"""
        load_path = path or self.storage_path

        with self.lock:
            if self.backend == StorageBackend.MEMORY:
                # Import from file for memory backend
                export_file = load_path / "knowledge_export.json"
                if export_file.exists():
                    self.import_from(export_file)
            elif self.backend == StorageBackend.SQLITE:
                # Load from database
                self._load_from_sqlite()
            elif self.backend == StorageBackend.FILE:
                # Load from individual files
                self._load_from_files()

            logger.info("Loaded knowledge base")

    def _initialize_backend(self):
        """Initialize storage backend"""
        if self.backend == StorageBackend.SQLITE:
            self._init_sqlite()
        elif self.backend == StorageBackend.FILE:
            self.storage_path.mkdir(parents=True, exist_ok=True)
        elif self.backend == StorageBackend.HYBRID:
            self._init_sqlite()  # Use SQLite for persistence
            # Keep everything in memory for speed

    def _init_sqlite(self):
        """Initialize SQLite backend with connection pool"""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        db_path = self.storage_path / "knowledge.db"

        # Create connection pool
        for i in range(self.pool_size):
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            self.conn_pool.append(conn)
            self.pool_available.append(i)

        # Create tables using first connection
        conn = self.conn_pool[0]
        conn.execute("""
            CREATE TABLE IF NOT EXISTS principles (
                id TEXT PRIMARY KEY,
                data BLOB,
                created_at REAL,
                updated_at REAL,
                access_count INTEGER DEFAULT 0,
                domain TEXT,
                confidence REAL
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_domain ON principles(domain)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_confidence ON principles(confidence)
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS versions (
                principle_id TEXT,
                version INTEGER,
                data BLOB,
                timestamp REAL,
                changes TEXT,
                author TEXT,
                message TEXT,
                PRIMARY KEY (principle_id, version),
                FOREIGN KEY (principle_id) REFERENCES principles(id)
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_author ON versions(author)
        """)

        conn.commit()

    def _persist_principle(self, principle_id: str, principle):
        """Persist principle to backend"""
        if principle is None:
            return

        if self.backend == StorageBackend.SQLITE:
            try:
                idx, conn = self._get_connection()
                try:
                    data = pickle.dumps(principle)

                    # Extract searchable fields
                    domain = getattr(principle, "domain", "general")
                    confidence = getattr(principle, "confidence", 0.5)

                    conn.execute(
                        """INSERT OR REPLACE INTO principles
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (
                            principle_id,
                            data,
                            self.creation_times.get(principle_id, time.time()),
                            self.update_times.get(principle_id, time.time()),
                            self.access_counts.get(principle_id, 0),
                            domain,
                            confidence,
                        ),
                    )
                    conn.commit()
                finally:
                    self._release_connection(idx)
            except Exception as e:
                logger.error("Failed to persist principle %s: %s", principle_id, e)
        elif self.backend == StorageBackend.FILE:
            try:
                path = self.storage_path / f"{principle_id}.pkl"
                with open(path, "wb") as f:
                    pickle.dump(principle, f)
            except Exception as e:
                logger.error(
                    "Failed to persist principle to file %s: %s", principle_id, e
                )

    def _persist_version(self, principle_id: str, version: PrincipleVersion):
        """Persist version to backend"""
        if self.backend == StorageBackend.SQLITE:
            try:
                idx, conn = self._get_connection()
                try:
                    # Store full principle or diff
                    if version.principle is not None:
                        data = pickle.dumps(version.principle)
                    else:
                        data = pickle.dumps(
                            {
                                "diff": version.principle_diff,
                                "base": version.base_version,
                            }
                        )

                    changes = json.dumps(version.changes)

                    conn.execute(
                        """INSERT OR REPLACE INTO versions
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (
                            principle_id,
                            version.version,
                            data,
                            version.timestamp,
                            changes,
                            version.author,
                            version.commit_message,
                        ),
                    )
                    conn.commit()
                finally:
                    self._release_connection(idx)
            except Exception as e:
                logger.error("Failed to persist version: %s", e)
        elif self.backend == StorageBackend.FILE:
            try:
                path = self.storage_path / f"{principle_id}_v{version.version}.pkl"
                with open(path, "wb") as f:
                    pickle.dump(version, f)
            except Exception as e:
                logger.error("Failed to persist version to file: %s", e)

    def _load_from_sqlite(self):
        """Load data from SQLite database"""
        try:
            idx, conn = self._get_connection()
            try:
                # Load principles
                cursor = conn.execute("""
                    SELECT id, data, created_at, updated_at, access_count
                    FROM principles
                """)

                for row in cursor:
                    principle_id, data, created_at, updated_at, access_count = row
                    principle = pickle.loads(data)
                    self.principles[principle_id] = principle
                    self.creation_times[principle_id] = created_at
                    self.update_times[principle_id] = updated_at
                    self.access_counts[principle_id] = access_count

                # Load versions
                cursor = conn.execute("""
                    SELECT principle_id, version, data, timestamp, changes, author, message
                    FROM versions
                    ORDER BY principle_id, version
                """)

                for row in cursor:
                    (
                        principle_id,
                        version_num,
                        data,
                        timestamp,
                        changes,
                        author,
                        message,
                    ) = row

                    try:
                        principle_data = pickle.loads(data)

                        # Check if it's a diff or full principle
                        if (
                            isinstance(principle_data, dict)
                            and "diff" in principle_data
                        ):
                            principle = None
                            principle_diff = principle_data["diff"]
                            base_version = principle_data["base"]
                        else:
                            principle = principle_data
                            principle_diff = None
                            base_version = None

                        changes_list = json.loads(changes) if changes else []

                        version = PrincipleVersion(
                            version=version_num,
                            principle=principle,
                            principle_diff=principle_diff,
                            base_version=base_version,
                            timestamp=timestamp,
                            changes=changes_list,
                            author=author,
                            commit_message=message,
                        )

                        self.versions[principle_id].append(version)
                        self.version_counters[principle_id] = max(
                            self.version_counters[principle_id], version_num
                        )
                    except Exception as e:
                        logger.error("Failed to load version: %s", e)

                self.total_principles = len(self.principles)
                self.total_versions = sum(len(v) for v in self.versions.values())
            finally:
                self._release_connection(idx)
        except Exception as e:
            logger.error("Failed to load from SQLite: %s", e)

    def _load_from_files(self):
        """Load data from individual files"""
        try:
            # Load principles
            for path in self.storage_path.glob("*.pkl"):
                if "_v" not in path.stem:  # Skip version files
                    try:
                        with open(path, "rb") as f:
                            principle = safe_pickle_load(f)
                            principle_id = path.stem
                            self.principles[principle_id] = principle
                    except Exception as e:
                        logger.error("Failed to load principle from %s: %s", path, e)

            # Load versions
            for path in self.storage_path.glob("*_v*.pkl"):
                parts = path.stem.split("_v")
                if len(parts) == 2:
                    principle_id = parts[0]
                    try:
                        version_num = int(parts[1])

                        with open(path, "rb") as f:
                            version_data = safe_pickle_load(f)

                        if isinstance(version_data, PrincipleVersion):
                            self.versions[principle_id].append(version_data)
                        else:
                            # Old format - create version object
                            version = PrincipleVersion(
                                version=version_num,
                                principle=version_data,
                                timestamp=path.stat().st_mtime,
                                changes=["Loaded from file"],
                            )
                            self.versions[principle_id].append(version)
                    except Exception as e:
                        logger.error("Failed to load version from %s: %s", path, e)

            self.total_principles = len(self.principles)
            self.total_versions = sum(len(v) for v in self.versions.values())
        except Exception as e:
            logger.error("Failed to load from files: %s", e)

    def _archive_principle(self, principle_id: str, principle):
        """Archive a principle before deletion"""
        try:
            archive_path = self.storage_path / "archive"
            archive_path.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_file = archive_path / f"{principle_id}_{timestamp}.pkl"

            with open(archive_file, "wb") as f:
                pickle.dump(
                    {
                        "principle": principle,
                        "versions": self.versions.get(principle_id, []),
                        "metadata": {
                            "archived_at": time.time(),
                            "creation_time": self.creation_times.get(principle_id),
                            "update_time": self.update_times.get(principle_id),
                            "access_count": self.access_counts.get(principle_id, 0),
                        },
                    },
                    f,
                )

            logger.debug("Archived principle %s to %s", principle_id, archive_file)
        except Exception as e:
            logger.error("Failed to archive principle %s: %s", principle_id, e)

    def _detect_changes(self, old_principle, new_principle) -> List[str]:
        """
        Detect changes between principles

        FIXED: Properly handles comparison by value, not reference
        """
        changes = []

        try:
            # Get attributes dictionaries
            old_attrs = (
                vars(old_principle) if hasattr(old_principle, "__dict__") else {}
            )
            new_attrs = (
                vars(new_principle) if hasattr(new_principle, "__dict__") else {}
            )

            # Track all keys
            all_keys = set(old_attrs.keys()) | set(new_attrs.keys())

            for key in all_keys:
                # Skip private attributes
                if key.startswith("_"):
                    continue

                # Check if attribute was added
                if key not in old_attrs:
                    changes.append(f"Added {key}")
                    continue

                # Check if attribute was removed
                if key not in new_attrs:
                    changes.append(f"Removed {key}")
                    continue

                # Compare values
                old_val = old_attrs[key]
                new_val = new_attrs[key]

                try:
                    # Use deep comparison
                    if self._values_differ(old_val, new_val):
                        changes.append(f"Modified {key}")
                except Exception as e:
                    # If comparison fails, assume changed
                    logger.debug(f"Error comparing {key}: {e}")
                    changes.append(f"Modified {key}")

        except Exception as e:
            logger.warning("Error detecting changes: %s", e)
            changes.append("Unknown changes")

        return changes

    def _values_differ(self, val1, val2) -> bool:
        """
        Compare two values for differences

        Helper method for _detect_changes
        """
        # Handle None
        if val1 is None and val2 is None:
            return False
        if val1 is None or val2 is None:
            return True

        # Type change is always a difference
        if type(val1) != type(val2):
            return True

        # Simple types - direct comparison
        if isinstance(val1, (str, int, float, bool)):
            return val1 != val2

        # Lists and tuples
        if isinstance(val1, (list, tuple)):
            if len(val1) != len(val2):
                return True
            return any(self._values_differ(v1, v2) for v1, v2 in zip(val1, val2))

        # Dictionaries
        if isinstance(val1, dict):
            if set(val1.keys()) != set(val2.keys()):
                return True
            return any(self._values_differ(val1[k], val2[k]) for k in val1.keys())

        # Sets
        if isinstance(val1, set):
            return val1 != val2

        # For other objects, try equality comparison
        try:
            return val1 != val2
        except Exception as e:  # If comparison fails, try string comparison
            return str(val1) != str(val2)

    def _calculate_similarity(self, p1, p2) -> float:
        """Calculate similarity between principles"""
        try:
            similarity = 0.0
            factors = 0

            # Domain similarity
            if hasattr(p1, "domain") and hasattr(p2, "domain"):
                if p1.domain == p2.domain:
                    similarity += 0.3
                elif self._domains_related(p1.domain, p2.domain):
                    similarity += 0.15
                factors += 0.3

            # Type similarity
            if hasattr(p1, "type") and hasattr(p2, "type"):
                if p1.type == p2.type:
                    similarity += 0.2
                factors += 0.2

            # Description similarity (simple word overlap)
            if hasattr(p1, "description") and hasattr(p2, "description"):
                words1 = set(str(p1.description).lower().split())
                words2 = set(str(p2.description).lower().split())
                if words1 and words2:
                    overlap = len(words1 & words2) / len(words1 | words2)
                    similarity += 0.5 * overlap
                factors += 0.5

            return similarity / factors if factors > 0 else 0.0
        except Exception as e:
            logger.warning("Error calculating similarity: %s", e)
            return 0.0

    def _domains_related(self, domain1: str, domain2: str) -> bool:
        """Check if domains are related"""
        try:
            if domain1 == domain2:
                return True

            # Check hierarchy
            parts1 = domain1.split("_")
            parts2 = domain2.split("_")

            # Common prefix indicates relationship
            for i in range(min(len(parts1), len(parts2))):
                if parts1[i] != parts2[i]:
                    return False
            return True
        except Exception:
            return False

    def _get_most_accessed(self, n: int) -> List[Tuple[str, int]]:
        """Get most accessed principles"""
        try:
            sorted_access = sorted(
                self.access_counts.items(), key=lambda x: x[1], reverse=True
            )
            return sorted_access[:n]
        except Exception:
            return []

    def __del__(self):
        """Cleanup on destruction"""
        # Close all database connections
        for conn in self.conn_pool:
            try:
                conn.close()
            except Exception as e:
                logger.debug(
                    f"{self.__class__.__name__ if hasattr(self, '__class__') else 'Operation'} error: {e}"
                )


class KnowledgeIndex:
    """Indexes knowledge for efficient retrieval"""

    def __init__(self, embedding_dim: int = 128):
        """
        Initialize knowledge index

        Args:
            embedding_dim: Dimension of embeddings
        """
        self.embedding_dim = embedding_dim

        # Indexes
        self.domain_index = defaultdict(set)  # domain -> set of principle_ids
        self.pattern_index = defaultdict(set)  # pattern -> set of principle_ids
        self.type_index = defaultdict(set)  # type -> set of principle_ids
        self.tag_index = defaultdict(set)  # tag -> set of principle_ids

        # Entries
        self.entries = {}  # principle_id -> IndexEntry

        # Vector index for similarity search
        self.vector_index = None
        self.indexed_ids = []
        self.embeddings = []

        # Thread safety
        self.lock = threading.RLock()

        # Statistics
        self.total_indexed = 0
        self.last_rebuild = time.time()

        logger.info("KnowledgeIndex initialized (FAISS available: %s)", FAISS_AVAILABLE)

    def index_principle(self, principle):
        """
        Fully index a principle

        Args:
            principle: Principle to index
        """
        with self.lock:
            if principle is None:
                return

            self.index_by_domain(principle)
            self.index_by_pattern(principle)
            self.index_by_type(principle)
            self.index_by_embedding(principle)

    def index_by_domain(self, principle):
        """
        Index principle by domain

        Args:
            principle: Principle to index
        """
        with self.lock:
            if principle is None:
                return

            principle_id = getattr(principle, "id", str(principle))
            domain = getattr(principle, "domain", "general")

            # Add to domain index
            self.domain_index[domain].add(principle_id)

            # Create or update entry
            if principle_id not in self.entries:
                self.entries[principle_id] = IndexEntry(
                    principle_id=principle_id, domain=domain, patterns=[]
                )
            else:
                self.entries[principle_id].domain = domain

            self.total_indexed += 1
            logger.debug("Indexed principle %s in domain %s", principle_id, domain)

    def index_by_pattern(self, principle):
        """
        Index principle by patterns

        Args:
            principle: Principle to index
        """
        with self.lock:
            if principle is None:
                return

            principle_id = getattr(principle, "id", str(principle))

            # Extract patterns
            patterns = []
            if hasattr(principle, "patterns"):
                patterns = principle.patterns
            elif hasattr(principle, "core_pattern"):
                # Extract from core pattern
                if hasattr(principle.core_pattern, "components"):
                    patterns = principle.core_pattern.components
            elif hasattr(principle, "description"):
                # Extract keywords as patterns
                patterns = self._extract_patterns(str(principle.description))

            # Add to pattern index
            for pattern in patterns:
                self.pattern_index[str(pattern)].add(principle_id)

            # Update entry
            if principle_id in self.entries:
                self.entries[principle_id].patterns = patterns
            else:
                self.entries[principle_id] = IndexEntry(
                    principle_id=principle_id, domain="general", patterns=patterns
                )

            logger.debug(
                "Indexed principle %s with %d patterns", principle_id, len(patterns)
            )

    def index_by_type(self, principle):
        """
        Index principle by type

        Args:
            principle: Principle to index
        """
        with self.lock:
            if principle is None:
                return

            if hasattr(principle, "type"):
                principle_id = getattr(principle, "id", str(principle))
                ptype = getattr(principle, "type")
                self.type_index[ptype].add(principle_id)

    def index_by_embedding(self, principle):
        """
        Index principle by embedding for similarity search

        Args:
            principle: Principle to index
        """
        with self.lock:
            if principle is None:
                return

            principle_id = getattr(principle, "id", str(principle))
            embedding = self._get_embedding(principle)

            if principle_id in self.entries:
                self.entries[principle_id].embedding = embedding
            else:
                self.entries[principle_id] = IndexEntry(
                    principle_id=principle_id,
                    domain="general",
                    patterns=[],
                    embedding=embedding,
                )

            # Mark for rebuild
            if time.time() - self.last_rebuild > 60:
                self._rebuild_vector_index()

    def find_relevant(self, problem: Dict[str, Any]) -> List[str]:
        """
        Find relevant principles for problem

        Args:
            problem: Problem specification

        Returns:
            List of principle IDs
        """
        with self.lock:
            if not problem:
                return []

            relevant_ids = set()

            # Search by domain
            if "domain" in problem:
                domain_matches = self.domain_index.get(problem["domain"], set())
                relevant_ids.update(domain_matches)

                # Also check parent domains
                if "_" in problem["domain"]:
                    parent = problem["domain"].split("_")[0]
                    parent_matches = self.domain_index.get(parent, set())
                    relevant_ids.update(parent_matches)

            # Search by patterns
            if "patterns" in problem:
                for pattern in problem["patterns"]:
                    pattern_matches = self.pattern_index.get(str(pattern), set())
                    relevant_ids.update(pattern_matches)

            # Search by keywords
            if "keywords" in problem:
                for keyword in problem["keywords"]:
                    # Check pattern index
                    keyword_matches = self.pattern_index.get(keyword, set())
                    relevant_ids.update(keyword_matches)

            # Search by type
            if "type" in problem:
                type_matches = self.type_index.get(problem["type"], set())
                relevant_ids.update(type_matches)

            # Search by tags
            if "tags" in problem:
                for tag in problem["tags"]:
                    tag_matches = self.tag_index.get(tag, set())
                    relevant_ids.update(tag_matches)

            # Score and rank
            scored = []
            for principle_id in relevant_ids:
                score = self._calculate_relevance_score(principle_id, problem)
                scored.append((principle_id, score))

            # Sort by score
            scored.sort(key=lambda x: x[1], reverse=True)

            # Update access statistics
            for pid, _ in scored:
                if pid in self.entries:
                    self.entries[pid].update_access()

            return [pid for pid, _ in scored]

    def search_by_similarity(
        self, pattern: Any, top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Search by similarity to pattern

        Args:
            pattern: Pattern to match (can be embedding or object)
            top_k: Number of results

        Returns:
            List of (principle_id, similarity_score) tuples
        """
        with self.lock:
            if pattern is None:
                return []

            # Get pattern embedding
            if isinstance(pattern, np.ndarray):
                query_embedding = pattern
            else:
                query_embedding = self._get_embedding(pattern)

            if self.vector_index is None or len(self.indexed_ids) == 0:
                self._rebuild_vector_index()

            if self.vector_index is None:
                logger.warning("Vector index not available")
                return []

            # Search
            k = min(max(1, top_k), len(self.indexed_ids))
            if k == 0:
                return []

            query_embedding = query_embedding.reshape(1, -1).astype("float32")
            distances, indices = self.vector_index.search(query_embedding, k)

            # Convert to results
            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if 0 <= idx < len(self.indexed_ids):
                    principle_id = self.indexed_ids[int(idx)]
                    similarity = 1.0 / (
                        1.0 + float(dist)
                    )  # Convert distance to similarity
                    results.append((principle_id, similarity))

                    # Update access
                    if principle_id in self.entries:
                        self.entries[principle_id].update_access()

            return results

    def update_index(self, principle):
        """
        Update all indexes for principle

        Args:
            principle: Principle to index
        """
        self.index_principle(principle)

    def remove_from_index(self, principle_id: str):
        """
        Remove principle from all indexes

        Args:
            principle_id: Principle ID to remove
        """
        with self.lock:
            if not principle_id:
                return

            # Remove from domain index
            if principle_id in self.entries:
                domain = self.entries[principle_id].domain
                self.domain_index[domain].discard(principle_id)

                # Remove from pattern index
                for pattern in self.entries[principle_id].patterns:
                    self.pattern_index[str(pattern)].discard(principle_id)

                # Remove entry
                del self.entries[principle_id]

            # Remove from type index
            for ptype, pids in list(self.type_index.items()):
                pids.discard(principle_id)
                if not pids:
                    del self.type_index[ptype]

            # Mark for rebuild
            self._rebuild_vector_index()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get index statistics

        Returns:
            Dictionary of statistics
        """
        with self.lock:
            # Access pattern analysis
            total_accesses = sum(e.access_count for e in self.entries.values())

            # Most accessed entries
            most_accessed = sorted(
                self.entries.items(), key=lambda x: x[1].access_count, reverse=True
            )[:10]

            return {
                "total_indexed": self.total_indexed,
                "domains": len(self.domain_index),
                "patterns": len(self.pattern_index),
                "types": len(self.type_index),
                "tags": len(self.tag_index),
                "entries": len(self.entries),
                "vector_index_size": len(self.indexed_ids),
                "total_accesses": total_accesses,
                "most_accessed": [(pid, e.access_count) for pid, e in most_accessed],
            }

    def _extract_patterns(self, text: str) -> List[str]:
        """Extract patterns from text"""
        try:
            # Simple keyword extraction
            words = text.lower().split()

            # Filter common words
            stop_words = {
                "the",
                "a",
                "an",
                "is",
                "are",
                "was",
                "were",
                "of",
                "to",
                "for",
                "in",
                "on",
                "at",
                "by",
                "with",
                "from",
                "up",
                "about",
                "into",
                "through",
                "during",
                "before",
                "after",
                "above",
                "below",
                "between",
            }

            keywords = list(words if w not in stop_words and len(w) > 3)

            # Extract bigrams
            bigrams = []
            for i in range(len(words) - 1):
                if words[i] not in stop_words and words[i + 1] not in stop_words:
                    bigrams.append(f"{words[i]}_{words[i + 1]}")

            return (keywords + bigrams)[:15]  # Limit total patterns
        except Exception as e:
            logger.warning("Failed to extract patterns: %s", e)
            return []

    def _calculate_relevance_score(
        self, principle_id: str, problem: Dict[str, Any]
    ) -> float:
        """Calculate relevance score"""
        try:
            score = 0.0

            if principle_id not in self.entries:
                return score

            entry = self.entries[principle_id]

            # Domain match
            if "domain" in problem:
                if entry.domain == problem["domain"]:
                    score += 0.4
                elif entry.domain == "general":
                    score += 0.1
                elif self._domains_related(entry.domain, problem["domain"]):
                    score += 0.2

            # Pattern overlap
            if "patterns" in problem and entry.patterns:
                problem_patterns = set(str(p) for p in problem["patterns"])
                entry_patterns = set(str(p) for p in entry.patterns)
                if problem_patterns and entry_patterns:
                    overlap = len(problem_patterns & entry_patterns)
                    score += 0.3 * (overlap / max(1, len(problem_patterns)))

            # Type match
            if "type" in problem and principle_id in self.entries:
                # Check if this principle has the matching type
                for ptype, pids in self.type_index.items():
                    if principle_id in pids and ptype == problem["type"]:
                        score += 0.2
                        break

            # Recency bonus
            age = time.time() - entry.timestamp
            recency_factor = np.exp(-age / (30 * 86400))  # 30-day decay
            score += 0.1 * recency_factor

            # Popularity bonus (based on access count)
            if entry.access_count > 0:
                popularity_factor = min(1.0, np.log1p(entry.access_count) / 10)
                score += 0.1 * popularity_factor

            return min(1.0, score)
        except Exception as e:
            logger.warning("Error calculating relevance: %s", e)
            return 0.0

    def _domains_related(self, domain1: str, domain2: str) -> bool:
        """Check if domains are related"""
        try:
            if domain1 == domain2:
                return True

            parts1 = domain1.split("_")
            parts2 = domain2.split("_")

            # Check for common parts
            return len(set(parts1) & set(parts2)) > 0
        except Exception:
            return False

    def _get_embedding(self, obj: Any) -> np.ndarray:
        """Get embedding for object"""
        try:
            # Enhanced embedding generation
            embedding = np.zeros(self.embedding_dim)

            # Combine multiple sources
            sources = []

            # Object string representation
            sources.append(str(obj))

            # Add specific attributes if available
            if hasattr(obj, "description"):
                sources.append(str(obj.description))
            if hasattr(obj, "domain"):
                sources.append(str(obj.domain))
            if hasattr(obj, "name"):
                sources.append(str(obj.name))

            # Combine sources
            combined = " ".join(sources)
            hash_val = hashlib.sha256(combined.encode()).hexdigest()

            # Convert hash to embedding
            for i in range(0, min(len(hash_val), self.embedding_dim * 2), 2):
                idx = (i // 2) % self.embedding_dim
                val = int(hash_val[i : i + 2], 16) / 255.0
                embedding[idx] = val

            # Add some randomness for diversity (using hash for determinism)
            np.random.seed(int(hash_val[:8], 16) % (2**32))
            noise = np.random.normal(0, 0.01, self.embedding_dim)
            embedding += noise

            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding /= norm

            return embedding.astype("float32")
        except Exception as e:
            logger.warning("Failed to get embedding: %s", e)
            return np.random.randn(self.embedding_dim).astype("float32")

    def _rebuild_vector_index(self):
        """Rebuild vector index"""
        try:
            if not self.entries:
                return

            # Collect embeddings
            self.indexed_ids = []
            embeddings = []

            for principle_id, entry in self.entries.items():
                if entry.embedding is not None:
                    self.indexed_ids.append(principle_id)
                    embeddings.append(entry.embedding)

            if not embeddings:
                return

            # Build index
            embeddings = np.array(embeddings).astype("float32")

            if FAISS_AVAILABLE:
                # Use FAISS if available
                if len(embeddings) > 1000:
                    # Use IVF index for large datasets
                    nlist = min(100, len(embeddings) // 10)
                    quantizer = faiss.IndexFlatL2(self.embedding_dim)
                    self.vector_index = faiss.IndexIVFFlat(
                        quantizer, self.embedding_dim, nlist
                    )
                    self.vector_index.train(embeddings)
                    self.vector_index.add(embeddings)
                else:
                    # Use flat index for small datasets
                    self.vector_index = faiss.IndexFlatL2(self.embedding_dim)
                    self.vector_index.add(embeddings)
            else:
                # Use simple fallback
                self.vector_index = SimpleVectorIndex(self.embedding_dim)
                self.vector_index.add(embeddings, self.indexed_ids)

            self.embeddings = embeddings
            self.last_rebuild = time.time()

            logger.debug("Rebuilt vector index with %d entries", len(self.indexed_ids))
        except Exception as e:
            logger.error("Failed to rebuild vector index: %s", e)


class KnowledgePruner:
    """Prunes outdated or low-quality knowledge"""

    def __init__(self, archive_path: Optional[Path] = None):
        """
        Initialize knowledge pruner

        Args:
            archive_path: Path for archiving pruned principles
        """
        self.pruned_count = 0
        self.archive = []
        self.archive_path = archive_path or Path("knowledge_archive")
        self.prune_history = deque(maxlen=1000)

        # Thread safety
        self.lock = threading.RLock()

        logger.info("KnowledgePruner initialized")

    def identify_outdated(
        self, knowledge_base: VersionedKnowledgeBase, age_threshold_days: int = 90
    ) -> List[PruneCandidate]:
        """
        Identify outdated principles

        Args:
            knowledge_base: Knowledge base to analyze
            age_threshold_days: Age threshold in days

        Returns:
            List of prune candidates
        """
        with self.lock:
            if not knowledge_base:
                return []

            candidates = []
            current_time = time.time()
            age_threshold_seconds = age_threshold_days * 86400

            for principle_id, update_time in knowledge_base.update_times.items():
                age = current_time - update_time

                if age > age_threshold_seconds:
                    # Check if frequently accessed
                    access_count = knowledge_base.access_counts.get(principle_id, 0)

                    # Check recent access
                    recent_accesses = knowledge_base.access_times.get(
                        principle_id, deque()
                    )
                    recent_access_count = sum(
                        1
                        for t in recent_accesses
                        if current_time - t < age_threshold_seconds
                    )

                    # Calculate prune score (higher = more likely to prune)
                    days_old = age / 86400
                    prune_score = (
                        (days_old / age_threshold_days)
                        * (1.0 / (1 + access_count))
                        * (1.0 / (1 + recent_access_count))
                    )

                    candidate = PruneCandidate(
                        principle_id=principle_id,
                        reason="outdated",
                        score=prune_score,
                        metadata={
                            "age_days": days_old,
                            "access_count": access_count,
                            "recent_access_count": recent_access_count,
                            "last_updated": update_time,
                        },
                    )
                    candidates.append(candidate)

            # Sort by prune score
            candidates.sort(key=lambda c: c.score, reverse=True)

            logger.info("Identified %d outdated principles", len(candidates))

            return candidates

    def identify_low_confidence(
        self, principles: List[Any], confidence_threshold: float = 0.3
    ) -> List[PruneCandidate]:
        """
        Identify low-confidence principles

        Args:
            principles: List of principles
            confidence_threshold: Confidence threshold

        Returns:
            List of prune candidates
        """
        with self.lock:
            if not principles:
                return []

            candidates = []

            for principle in principles:
                if principle is None:
                    continue

                if hasattr(principle, "confidence"):
                    if principle.confidence < confidence_threshold:
                        # Calculate prune score
                        prune_score = 1.0 - principle.confidence

                        # Consider success/failure ratio
                        if hasattr(principle, "success_count") and hasattr(
                            principle, "failure_count"
                        ):
                            total = principle.success_count + principle.failure_count
                            if total > 0:
                                success_rate = principle.success_count / total
                                prune_score *= 1.0 - success_rate

                        candidate = PruneCandidate(
                            principle_id=getattr(principle, "id", str(principle)),
                            reason="low_confidence",
                            score=prune_score,
                            metadata={
                                "confidence": principle.confidence,
                                "success_count": getattr(principle, "success_count", 0),
                                "failure_count": getattr(principle, "failure_count", 0),
                            },
                        )
                        candidates.append(candidate)

            candidates.sort(key=lambda c: c.score, reverse=True)

            logger.info("Identified %d low-confidence principles", len(candidates))

            return candidates

    def prune_contradictory(self, principles: List[Any]) -> List[PruneCandidate]:
        """
        Identify contradictory principles

        Args:
            principles: List of principles

        Returns:
            List of prune candidates
        """
        with self.lock:
            if not principles or len(principles) < 2:
                return []

            candidates = []

            # Group by domain and type
            groups = defaultdict(list)
            for principle in principles:
                if principle is None:
                    continue

                key = (
                    getattr(principle, "domain", "general"),
                    getattr(principle, "type", "general"),
                )
                groups[key].append(principle)

            # Find contradictions within groups
            for (domain, ptype), group_principles in groups.items():
                if len(group_principles) < 2:
                    continue

                # Check for contradictions
                for i, p1 in enumerate(group_principles):
                    for p2 in group_principles[i + 1 :]:
                        if self._are_contradictory(p1, p2):
                            # Keep the one with higher confidence and success rate
                            score1 = self._calculate_principle_score(p1)
                            score2 = self._calculate_principle_score(p2)

                            to_prune = p1 if score1 < score2 else p2

                            candidate = PruneCandidate(
                                principle_id=getattr(to_prune, "id", str(to_prune)),
                                reason="contradictory",
                                score=1.0 - self._calculate_principle_score(to_prune),
                                metadata={
                                    "contradicts": getattr(
                                        p2 if to_prune == p1 else p1, "id", "unknown"
                                    ),
                                    "domain": domain,
                                    "type": ptype,
                                },
                            )
                            candidates.append(candidate)

            # Remove duplicates
            seen = set()
            unique_candidates = []
            for candidate in candidates:
                if candidate.principle_id not in seen:
                    seen.add(candidate.principle_id)
                    unique_candidates.append(candidate)

            unique_candidates.sort(key=lambda c: c.score, reverse=True)

            logger.info(
                "Identified %d contradictory principles", len(unique_candidates)
            )

            return unique_candidates

    def archive_pruned(self, principle):
        """
        Archive pruned principle

        Args:
            principle: Principle to archive
        """
        with self.lock:
            if principle is None:
                return

            try:
                archived_entry = {
                    "principle": copy.deepcopy(principle),
                    "archived_at": time.time(),
                    "principle_id": getattr(principle, "id", str(principle)),
                }

                self.archive.append(archived_entry)
                self.pruned_count += 1

                # Save to file if path specified
                if self.archive_path:
                    self.archive_path.mkdir(parents=True, exist_ok=True)
                    archive_file = (
                        self.archive_path
                        / f"{archived_entry['principle_id']}_{int(time.time())}.pkl"
                    )
                    with open(archive_file, "wb") as f:
                        pickle.dump(archived_entry, f)

                # Track in history
                self.prune_history.append(
                    {
                        "principle_id": archived_entry["principle_id"],
                        "timestamp": archived_entry["archived_at"],
                        "reason": "manual_archive",
                    }
                )

                logger.debug("Archived principle %s", archived_entry["principle_id"])
            except Exception as e:
                logger.error("Failed to archive principle: %s", e)

    def execute_pruning(
        self,
        candidates: List[PruneCandidate],
        knowledge_base: VersionedKnowledgeBase,
        threshold: float = 0.7,
    ) -> int:
        """
        Execute pruning based on candidates

        Args:
            candidates: Prune candidates
            knowledge_base: Knowledge base
            threshold: Score threshold for pruning

        Returns:
            Number of pruned principles
        """
        with self.lock:
            if not candidates or not knowledge_base:
                return 0

            pruned = 0

            for candidate in candidates:
                if candidate.score >= threshold:
                    principle = knowledge_base.get(candidate.principle_id)

                    if principle:
                        # Archive before removing
                        self.archive_pruned(principle)

                        # Remove from knowledge base
                        if knowledge_base.delete(candidate.principle_id, soft=True):
                            pruned += 1

                            # Track in history
                            self.prune_history.append(
                                {
                                    "principle_id": candidate.principle_id,
                                    "timestamp": time.time(),
                                    "reason": candidate.reason,
                                    "score": candidate.score,
                                }
                            )

            logger.info("Pruned %d principles", pruned)

            return pruned

    def restore_from_archive(
        self, principle_id: str, knowledge_base: VersionedKnowledgeBase
    ) -> bool:
        """
        Restore principle from archive

        Args:
            principle_id: Principle ID to restore
            knowledge_base: Knowledge base to restore to

        Returns:
            True if successful
        """
        with self.lock:
            if not principle_id or not knowledge_base:
                return False

            # Search in memory archive
            for entry in self.archive:
                if entry["principle_id"] == principle_id:
                    knowledge_base.store(
                        entry["principle"],
                        author="system",
                        message="Restored from archive",
                    )
                    logger.info("Restored principle %s from archive", principle_id)
                    return True

            # Search in file archive
            if self.archive_path and self.archive_path.exists():
                for archive_file in self.archive_path.glob(f"{principle_id}_*.pkl"):
                    try:
                        with open(archive_file, "rb") as f:
                            entry = safe_pickle_load(f)

                        knowledge_base.store(
                            entry["principle"],
                            author="system",
                            message="Restored from archive",
                        )
                        logger.info(
                            "Restored principle %s from file archive", principle_id
                        )
                        return True
                    except Exception as e:
                        logger.error("Failed to restore from %s: %s", archive_file, e)

            logger.warning("Principle %s not found in archive", principle_id)
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get pruning statistics

        Returns:
            Dictionary of statistics
        """
        with self.lock:
            reason_counts = defaultdict(int)
            for entry in self.prune_history:
                reason_counts[entry.get("reason", "unknown")] += 1

            return {
                "total_pruned": self.pruned_count,
                "archived_count": len(self.archive),
                "prune_reasons": dict(reason_counts),
                "recent_prunes": len(self.prune_history),
                "archive_path": str(self.archive_path) if self.archive_path else None,
            }

    def _are_contradictory(self, p1: Any, p2: Any) -> bool:
        """Check if two principles are contradictory"""
        try:
            if p1 is None or p2 is None:
                return False

            # Check descriptions if available
            if hasattr(p1, "description") and hasattr(p2, "description"):
                desc1 = str(p1.description).lower()
                desc2 = str(p2.description).lower()

                # Check for negation patterns
                negation_pairs = [
                    ("always", "never"),
                    ("increase", "decrease"),
                    ("maximize", "minimize"),
                    ("true", "false"),
                    ("positive", "negative"),
                    ("enable", "disable"),
                    ("allow", "prevent"),
                    ("include", "exclude"),
                ]

                for word1, word2 in negation_pairs:
                    if (word1 in desc1 and word2 in desc2) or (
                        word2 in desc1 and word1 in desc2
                    ):
                        return True

            # Check parameters if available
            if hasattr(p1, "parameters") and hasattr(p2, "parameters"):
                if isinstance(p1.parameters, dict) and isinstance(p2.parameters, dict):
                    for key in p1.parameters:
                        if key in p2.parameters:
                            v1 = p1.parameters[key]
                            v2 = p2.parameters[key]

                            # Check for opposite values
                            if isinstance(v1, bool) and isinstance(v2, bool):
                                if v1 != v2:
                                    return True
                            elif isinstance(v1, (int, float)) and isinstance(
                                v2, (int, float)
                            ):
                                # Check for significant opposite values
                                if (v1 > 0 and v2 < 0) or (v1 < 0 and v2 > 0):
                                    return True

            return False
        except Exception as e:
            logger.warning("Error checking contradiction: %s", e)
            return False

    def _calculate_principle_score(self, principle) -> float:
        """Calculate overall score for a principle"""
        try:
            if principle is None:
                return 0.0

            score = 0.0

            # Confidence component
            if hasattr(principle, "confidence"):
                score += principle.confidence * 0.5

            # Success rate component
            if hasattr(principle, "success_count") and hasattr(
                principle, "failure_count"
            ):
                total = principle.success_count + principle.failure_count
                if total > 0:
                    success_rate = principle.success_count / total
                    score += success_rate * 0.3

            # Recency component
            if hasattr(principle, "last_updated"):
                age_days = (time.time() - principle.last_updated) / 86400
                recency_score = np.exp(-age_days / 30)  # 30-day half-life
                score += recency_score * 0.2

            return min(1.0, score)
        except Exception as e:
            logger.warning("Error calculating principle score: %s", e)
            return 0.0
