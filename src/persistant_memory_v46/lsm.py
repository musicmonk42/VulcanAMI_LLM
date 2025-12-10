from __future__ import annotations

import asyncio
import hashlib
import logging
import pickle
import time
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class BloomFilter:
    """Space-efficient probabilistic data structure for membership testing."""

    def __init__(self, size: int = 10000, num_hashes: int = 3):
        self.size = size
        self.num_hashes = num_hashes
        self.bit_array = np.zeros(size, dtype=bool)
        self.item_count = 0

    def _hashes(self, item: str) -> List[int]:
        """Generate multiple hash values for an item."""
        hashes = []
        for i in range(self.num_hashes):
            hash_input = f"{item}:{i}".encode()
            hash_val = int(hashlib.sha256(hash_input).hexdigest()[:16], 16)
            hashes.append(hash_val % self.size)
        return hashes

    def add(self, item: str) -> None:
        """Add an item to the bloom filter."""
        for h in self._hashes(item):
            self.bit_array[h] = True
        self.item_count += 1

    def contains(self, item: str) -> bool:
        """Check if an item might be in the set (no false negatives)."""
        return all(self.bit_array[h] for h in self._hashes(item))

    def false_positive_rate(self) -> float:
        """Estimate current false positive rate."""
        if self.item_count == 0:
            return 0.0

        # (1 - e^(-kn/m))^k where k=num_hashes, n=items, m=size
        k = self.num_hashes
        n = self.item_count
        m = self.size

        return (1 - np.exp(-k * n / m)) ** k

    def serialize(self) -> bytes:
        """Serialize bloom filter to bytes."""
        data = {
            "size": self.size,
            "num_hashes": self.num_hashes,
            "bit_array": self.bit_array.tobytes(),
            "item_count": self.item_count,
        }
        return pickle.dumps(data)

    @classmethod
    def deserialize(cls, data: bytes) -> BloomFilter:
        """Deserialize bloom filter from bytes."""
        obj_data = pickle.loads(data)
        bf = cls(obj_data["size"], obj_data["num_hashes"])
        bf.bit_array = np.frombuffer(obj_data["bit_array"], dtype=bool)
        bf.item_count = obj_data["item_count"]
        return bf


@dataclass
class Packfile:
    """Represents a packfile in the LSM tree."""

    pack_id: str
    level: int
    data: bytes
    bloom_filter: Optional[BloomFilter] = None
    min_key: Optional[str] = None
    max_key: Optional[str] = None
    item_count: int = 0
    size_bytes: int = 0
    created_at: float = field(default_factory=time.time)
    checksum: Optional[str] = None

    def __post_init__(self):
        if self.checksum is None:
            self.checksum = hashlib.sha256(self.data).hexdigest()


@dataclass
class MerkleNode:
    """Node in the Merkle DAG."""

    node_hash: str
    parent_hashes: List[str] = field(default_factory=list)
    pack_ids: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MerkleLSMDAG:
    """Merkle DAG for tracking LSM tree history and lineage."""

    def __init__(self):
        self.nodes: Dict[str, MerkleNode] = {}
        self.head: Optional[str] = None
        self.branches: Dict[str, str] = {"main": None}

    def add_node(
        self,
        pack_ids: List[str],
        parent_hashes: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add a new node to the DAG."""
        # Compute node hash
        content = f"{pack_ids}:{parent_hashes}:{time.time()}".encode()
        node_hash = hashlib.sha256(content).hexdigest()

        node = MerkleNode(
            node_hash=node_hash,
            parent_hashes=parent_hashes or [],
            pack_ids=pack_ids,
            metadata=metadata or {},
        )

        self.nodes[node_hash] = node
        self.head = node_hash
        self.branches["main"] = node_hash

        return node_hash

    def get_lineage(self, node_hash: str) -> List[str]:
        """Get the lineage of a node back to root."""
        lineage = []
        current = node_hash
        visited = set()

        while current and current not in visited:
            lineage.append(current)
            visited.add(current)

            node = self.nodes.get(current)
            if node and node.parent_hashes:
                current = node.parent_hashes[0]
            else:
                current = None

        return lineage

    def verify_integrity(self, node_hash: str) -> bool:
        """Verify the integrity of a node and its parents."""
        if node_hash not in self.nodes:
            return False

        node = self.nodes[node_hash]

        # Verify parents exist
        for parent_hash in node.parent_hashes:
            if parent_hash not in self.nodes:
                return False

        return True

    def get_dag_stats(self) -> Dict[str, Any]:
        """Get DAG statistics."""
        return {
            "total_nodes": len(self.nodes),
            "head": self.head,
            "branches": self.branches,
            "max_depth": self._compute_max_depth(),
        }

    def _compute_max_depth(self) -> int:
        """Compute maximum depth of the DAG."""
        if not self.head:
            return 0

        return len(self.get_lineage(self.head))


@dataclass
class MerkleLSM:
    """
    Merkle Log-Structured Merge Tree with advanced features.

    Features:
    - Multi-level compaction
    - Bloom filters for fast negative lookups
    - Adaptive compaction strategies
    - Background compaction
    - Merkle DAG for versioning
    - Point and range queries
    - Pattern matching with regex
    - Snapshot isolation
    """

    packfile_size_mb: int = 32
    compaction_strategy: str = "adaptive"
    bloom_filter: bool = True
    bloom_size: int = 100000
    bloom_hashes: int = 3
    max_levels: int = 7
    level_multiplier: int = 10
    compaction_trigger_ratio: float = 4.0
    background_compaction: bool = True
    compression: str = "zlib"

    def __post_init__(self):
        """Initialize the MerkleLSM."""
        self.dag = MerkleLSMDAG()
        self.memtable: Dict[str, Any] = {}
        self.packfiles: Dict[int, List[Packfile]] = defaultdict(list)
        self.wal: List[Tuple[str, str, Any]] = []  # Write-ahead log
        self.compaction_stats: Dict[str, Any] = defaultdict(int)
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.is_compacting = False
        self.snapshots: Dict[str, Dict[str, Any]] = {}

        # Start background compaction if enabled
        if self.background_compaction:
            self._start_background_compaction()

        logger.info(f"MerkleLSM initialized with strategy={self.compaction_strategy}")

    def put(self, key: str, value: Any) -> None:
        """Write a key-value pair to the LSM tree."""
        # Write to WAL first for durability
        self.wal.append(("PUT", key, value))

        # Write to memtable
        self.memtable[key] = value

        # Flush to disk if memtable is too large
        memtable_size = self._estimate_memtable_size()
        if memtable_size >= self.packfile_size_mb * 1024 * 1024:
            self.flush_memtable()

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a value by key."""
        # Check memtable first
        if key in self.memtable:
            value = self.memtable[key]
            # Return None if tombstone
            if self._is_tombstone(value):
                return None
            return value

        # Check packfiles from newest to oldest
        for level in sorted(self.packfiles.keys()):
            for packfile in reversed(self.packfiles[level]):
                # Use bloom filter if available
                if packfile.bloom_filter and not packfile.bloom_filter.contains(key):
                    continue

                # Check key range
                if packfile.min_key and packfile.max_key:
                    if key < packfile.min_key or key > packfile.max_key:
                        continue

                # Search in packfile
                value = self._search_packfile(packfile, key)
                if value is not None:
                    # Return None if tombstone
                    if self._is_tombstone(value):
                        return None
                    return value

        return None

    def delete(self, key: str) -> None:
        """Delete a key (using tombstone)."""
        # WAL write
        self.wal.append(("DELETE", key, None))

        # Mark as deleted with tombstone
        self.memtable[key] = ("__TOMBSTONE__", time.time())

    def flush_memtable(self) -> str:
        """Flush memtable to a Level 0 packfile."""
        if not self.memtable:
            return ""

        logger.info(f"Flushing memtable with {len(self.memtable)} entries")

        # Create packfile
        pack_id = f"pack-{int(time.time() * 1000)}-L0"
        pack_data = self._serialize_memtable()

        # Create bloom filter if enabled
        bloom = None
        if self.bloom_filter:
            bloom = BloomFilter(self.bloom_size, self.bloom_hashes)
            for key in self.memtable.keys():
                bloom.add(key)

        # Get key range
        keys = sorted(self.memtable.keys())
        min_key = keys[0] if keys else None
        max_key = keys[-1] if keys else None

        # Create packfile object
        packfile = Packfile(
            pack_id=pack_id,
            level=0,
            data=pack_data,
            bloom_filter=bloom,
            min_key=min_key,
            max_key=max_key,
            item_count=len(self.memtable),
            size_bytes=len(pack_data),
        )

        self.packfiles[0].append(packfile)

        # Update DAG
        self.dag.add_node(
            pack_ids=[pack_id],
            metadata={"action": "flush", "level": 0, "items": len(self.memtable)},
        )

        # Clear memtable and WAL
        self.memtable.clear()
        self.wal.clear()

        # Trigger compaction if needed
        if self._should_compact(0):
            self.compact_level(0)

        return pack_id

    def compact(self, items: Iterable[Any]) -> bytes:
        """
        Compact items into a packfile.

        Args:
            items: Iterable of items to compact

        Returns:
            Packfile bytes
        """
        # Convert items to sorted key-value pairs
        item_dict = {}
        for item in items:
            if isinstance(item, tuple) and len(item) == 2:
                key, value = item
                item_dict[key] = value
            elif isinstance(item, dict):
                item_dict.update(item)

        # Sort by key
        sorted_items = sorted(item_dict.items())

        # Serialize
        serialized = pickle.dumps(sorted_items)

        # Compress if enabled
        if self.compression == "zlib":
            serialized = zlib.compress(serialized, level=6)
        elif self.compression == "zstd":
            try:
                import zstandard as zstd

                compressor = zstd.ZstdCompressor(level=3)
                serialized = compressor.compress(serialized)
            except ImportError:
                logger.warning("zstd not available, using zlib")
                serialized = zlib.compress(serialized, level=6)

        return serialized

    def compact_level(self, level: int) -> List[str]:
        """
        Compact packfiles at a specific level.

        Args:
            level: Level to compact

        Returns:
            List of new packfile IDs
        """
        if level not in self.packfiles or not self.packfiles[level]:
            return []

        self.is_compacting = True
        start_time = time.time()

        logger.info(
            f"Compacting level {level} with {len(self.packfiles[level])} packfiles"
        )

        try:
            # Select packfiles to compact based on strategy
            if self.compaction_strategy == "tiered":
                packs_to_compact = self.packfiles[level]
            elif self.compaction_strategy == "leveled":
                packs_to_compact = self._select_leveled_compaction(level)
            elif self.compaction_strategy == "adaptive":
                packs_to_compact = self._select_adaptive_compaction(level)
            else:
                packs_to_compact = self.packfiles[level]

            if not packs_to_compact:
                return []

            # Merge packfiles
            merged_data = self._merge_packfiles(packs_to_compact)

            # Split into new packfiles for next level
            new_packfiles = self._split_compacted_data(merged_data, level + 1)

            # Remove old packfiles
            for pack in packs_to_compact:
                self.packfiles[level].remove(pack)

            # Add new packfiles to next level
            new_pack_ids = []
            for packfile in new_packfiles:
                self.packfiles[level + 1].append(packfile)
                new_pack_ids.append(packfile.pack_id)

            # Update DAG
            old_pack_ids = [p.pack_id for p in packs_to_compact]
            self.dag.add_node(
                pack_ids=new_pack_ids,
                metadata={
                    "action": "compact",
                    "from_level": level,
                    "to_level": level + 1,
                    "old_packs": old_pack_ids,
                    "strategy": self.compaction_strategy,
                },
            )

            # Update stats
            self.compaction_stats["total_compactions"] += 1
            self.compaction_stats[f"level_{level}_compactions"] += 1

            elapsed = time.time() - start_time
            logger.info(
                f"Level {level} compaction completed in {elapsed:.2f}s, created {len(new_packfiles)} packfiles"
            )

            return new_pack_ids

        finally:
            self.is_compacting = False

    def compact_background(self) -> None:
        """Run background compaction on all levels."""
        if self.is_compacting:
            logger.debug("Compaction already in progress, skipping")
            return

        for level in range(self.max_levels - 1):
            if self._should_compact(level):
                self.compact_level(level)

    def find_pattern(self, pattern: str) -> List[str]:
        """
        Search for keys matching a pattern using pack-level bloom filters.

        Args:
            pattern: Pattern to search for (supports * wildcard)

        Returns:
            List of matching keys
        """
        import re

        # Convert wildcard pattern to regex
        regex_pattern = pattern.replace("*", ".*")
        regex = re.compile(regex_pattern)

        matching_keys = list(]

        # Search memtable
        for key in self.memtable.keys():
            if regex.match(key):
                matching_keys.append(key)

        # Search packfiles
        for level in sorted(self.packfiles.keys()):
            for packfile in self.packfiles[level]:
                # Use bloom filter for prefix patterns
                if pattern.endswith("*"):
                    prefix = pattern[:-1]
                    if packfile.bloom_filter and not packfile.bloom_filter.contains(
                        prefix
                    ):
                        continue

                # Search packfile contents
                keys = self._get_packfile_keys(packfile)
                for key in keys:
                    if regex.match(key) and key not in matching_keys:
                        matching_keys.append(key)

        return sorted(matching_keys)

    def pattern_match(self, pattern: str) -> List[Tuple[str, Any]]:
        """
        Find all key-value pairs matching a regex pattern.

        Args:
            pattern: Regex pattern to match keys against

        Returns:
            List of (key, value) tuples matching the pattern
        """
        import re

        regex = re.compile(pattern)
        results = []
        seen_keys = set()

        # Search memtable
        for key, value in self.memtable.items():
            if regex.match(key) and not self._is_tombstone(value):
                results.append((key, value))
                seen_keys.add(key)

        # Search packfiles
        for level in range(self.max_levels):
            if level not in self.packfiles:
                continue

            for packfile in self.packfiles[level]:
                keys = self._get_packfile_keys(packfile)
                for key in keys:
                    if key in seen_keys:
                        continue
                    if regex.match(key):
                        value = self._search_packfile(packfile, key)
                        if value is not None and not self._is_tombstone(value):
                            results.append((key, value))
                            seen_keys.add(key)

        return sorted(results)

    def range_query(self, start_key: str, end_key: str) -> List[Tuple[str, Any]]:
        """
        Perform a range query.

        Args:
            start_key: Start of range (inclusive)
            end_key: End of range (exclusive)

        Returns:
            List of (key, value) pairs in range
        """
        results = list(]
        seen_keys = set()

        # Search memtable
        for key, value in self.memtable.items():
            if start_key <= key < end_key:
                results.append((key, value))
                seen_keys.add(key)

        # Search packfiles from newest to oldest
        for level in sorted(self.packfiles.keys()):
            for packfile in reversed(self.packfiles[level]):
                # Check if range overlaps with packfile range
                if packfile.min_key and packfile.max_key:
                    if end_key <= packfile.min_key or start_key >= packfile.max_key:
                        continue

                # Get all entries in range from packfile
                entries = self._range_query_packfile(packfile, start_key, end_key)
                for key, value in entries:
                    if key not in seen_keys:
                        results.append((key, value))
                        seen_keys.add(key)

        # Sort and filter tombstones
        results.sort(key=lambda x: x[0])
        results = [(k, v) for k, v in results if not self._is_tombstone(v)]

        return results

    def create_snapshot(self, snapshot_id: Optional[str] = None) -> str:
        """
        Create a snapshot of current state.

        Args:
            snapshot_id: Optional snapshot ID (auto-generated if not provided)

        Returns:
            The snapshot ID
        """
        if snapshot_id is None:
            snapshot_id = f"snapshot-{int(time.time() * 1000)}"

        self.snapshots[snapshot_id] = {
            "memtable": dict(self.memtable),
            "packfiles": {
                level: list(packs) for level, packs in self.packfiles.items()
            },
            "timestamp": time.time(),
        }

        logger.info(f"Created snapshot {snapshot_id}")
        return snapshot_id

    def restore_snapshot(self, snapshot_id: str) -> None:
        """
        Restore LSM tree to a previous snapshot state.

        Args:
            snapshot_id: ID of snapshot to restore
        """
        if snapshot_id not in self.snapshots:
            raise ValueError(f"Snapshot {snapshot_id} not found")

        snapshot = self.snapshots[snapshot_id]

        # Restore memtable
        self.memtable = dict(snapshot["memtable"])

        # Restore packfiles (deep copy)
        self.packfiles.clear()
        for level, packs in snapshot["packfiles"].items():
            self.packfiles[level] = list(packs)

        logger.info(f"Restored snapshot {snapshot_id}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get LSM tree statistics."""
        total_packfiles = sum(len(packs) for packs in self.packfiles.values())
        total_size = sum(
            sum(p.size_bytes for p in packs) for packs in self.packfiles.values()
        )

        # Calculate total items across all levels
        total_items = sum(
            sum(p.item_count for p in packs) for packs in self.packfiles.values()
        )

        level_stats = {}
        for level, packs in self.packfiles.items():
            level_stats[f"level_{level}"] = {
                "packfiles": len(packs),
                "total_size_mb": sum(p.size_bytes for p in packs) / (1024 * 1024),
                "total_items": sum(p.item_count for p in packs),
            }

        return {
            "memtable_size": len(self.memtable),
            "total_packfiles": total_packfiles,
            "total_items": total_items,  # Added for test compatibility
            "total_size_mb": total_size / (1024 * 1024),
            "compaction_stats": dict(self.compaction_stats),
            "level_stats": level_stats,
            "dag_stats": self.dag.get_dag_stats(),
            "wal_size": len(self.wal),
            "snapshots": len(self.snapshots),
            "is_compacting": self.is_compacting,
        }

    def _estimate_memtable_size(self) -> int:
        """Estimate memtable size in bytes."""
        # Rough estimate
        return sum(len(str(k)) + len(pickle.dumps(v)) for k, v in self.memtable.items())

    def _serialize_memtable(self) -> bytes:
        """Serialize memtable to bytes."""
        sorted_items = sorted(self.memtable.items())
        serialized = pickle.dumps(sorted_items)

        if self.compression == "zlib":
            return zlib.compress(serialized, level=6)

        return serialized

    def _search_packfile(self, packfile: Packfile, key: str) -> Optional[Any]:
        """Search for a key in a packfile."""
        try:
            # Decompress
            data = packfile.data
            if self.compression == "zlib":
                data = zlib.decompress(data)

            # Deserialize
            items = pickle.loads(data)

            # Binary search (items are sorted)
            left, right = 0, len(items) - 1

            while left <= right:
                mid = (left + right) // 2
                mid_key, mid_value = items[mid]

                if mid_key == key:
                    return mid_value
                elif mid_key < key:
                    left = mid + 1
                else:
                    right = mid - 1

            return None

        except Exception as e:
            logger.error(f"Error searching packfile {packfile.pack_id}: {e}")
            return None

    def _get_packfile_keys(self, packfile: Packfile) -> List[str]:
        """Get all keys from a packfile."""
        try:
            data = packfile.data
            if self.compression == "zlib":
                data = zlib.decompress(data)

            items = pickle.loads(data)
            return [k for k, v in items]

        except Exception as e:
            logger.error(f"Error getting keys from packfile {packfile.pack_id}: {e}")
            return []

    def _range_query_packfile(
        self, packfile: Packfile, start_key: str, end_key: str
    ) -> List[Tuple[str, Any]]:
        """Perform range query on a packfile."""
        try:
            data = packfile.data
            if self.compression == "zlib":
                data = zlib.decompress(data)

            items = pickle.loads(data)

            # Binary search for start
            left = 0
            right = len(items) - 1
            start_idx = 0

            while left <= right:
                mid = (left + right) // 2
                if items[mid][0] < start_key:
                    left = mid + 1
                    start_idx = left
                else:
                    right = mid - 1

            # Collect items in range
            results = []
            for i in range(start_idx, len(items)):
                key, value = items[i]
                if key >= end_key:
                    break
                results.append((key, value))

            return results

        except Exception as e:
            logger.error(f"Error in range query on packfile {packfile.pack_id}: {e}")
            return []

    def _merge_packfiles(self, packfiles: List[Packfile]) -> Dict[str, Any]:
        """Merge multiple packfiles."""
        merged = {}

        # Merge all packfiles (newer values override older)
        for packfile in packfiles:
            try:
                data = packfile.data
                if self.compression == "zlib":
                    data = zlib.decompress(data)

                items = pickle.loads(data)
                for key, value in items:
                    if not self._is_tombstone(value):
                        merged[key] = value

            except Exception as e:
                logger.error(f"Error merging packfile {packfile.pack_id}: {e}")

        return merged

    def _split_compacted_data(self, data: Dict[str, Any], level: int) -> List[Packfile]:
        """Split compacted data into multiple packfiles."""
        sorted_items = sorted(data.items())

        # Calculate target size
        target_size = self.packfile_size_mb * 1024 * 1024

        packfiles = []
        current_items = []
        current_size = 0

        for key, value in sorted_items:
            item_size = len(str(key)) + len(pickle.dumps(value))

            if current_size + item_size > target_size and current_items:
                # Create packfile from current items
                packfile = self._create_packfile(current_items, level)
                packfiles.append(packfile)

                current_items = []
                current_size = 0

            current_items.append((key, value))
            current_size += item_size

        # Create last packfile
        if current_items:
            packfile = self._create_packfile(current_items, level)
            packfiles.append(packfile)

        return packfiles

    def _create_packfile(self, items: List[Tuple[str, Any]], level: int) -> Packfile:
        """Create a packfile from items."""
        pack_id = f"pack-{int(time.time() * 1000)}-L{level}"

        # Serialize
        serialized = pickle.dumps(items)
        if self.compression == "zlib":
            serialized = zlib.compress(serialized, level=6)

        # Create bloom filter
        bloom = None
        if self.bloom_filter:
            bloom = BloomFilter(self.bloom_size, self.bloom_hashes)
            for key, _ in items:
                bloom.add(key)

        # Get key range
        min_key = items[0][0] if items else None
        max_key = items[-1][0] if items else None

        return Packfile(
            pack_id=pack_id,
            level=level,
            data=serialized,
            bloom_filter=bloom,
            min_key=min_key,
            max_key=max_key,
            item_count=len(items),
            size_bytes=len(serialized),
        )

    def _should_compact(self, level: int) -> bool:
        """Check if a level should be compacted."""
        if level >= self.max_levels - 1:
            return False

        if level not in self.packfiles:
            return False

        current_count = len(self.packfiles[level])

        # Level 0 has fixed trigger
        if level == 0:
            return current_count >= 4

        # Other levels use ratio
        target_count = self.level_multiplier**level
        return current_count >= target_count * self.compaction_trigger_ratio

    def _select_leveled_compaction(self, level: int) -> List[Packfile]:
        """Select packfiles for leveled compaction."""
        # Select all packfiles at this level
        return self.packfiles[level]

    def _select_adaptive_compaction(self, level: int) -> List[Packfile]:
        """Select packfiles for adaptive compaction."""
        packs = self.packfiles[level]

        if len(packs) <= 2:
            return packs

        # Select packfiles with most overlap
        # For simplicity, select oldest packfiles
        sorted_packs = sorted(packs, key=lambda p: p.created_at)

        return sorted_packs[: max(2, len(packs) // 2)]

    def _is_tombstone(self, value: Any) -> bool:
        """Check if a value is a tombstone."""
        return (
            isinstance(value, tuple) and len(value) == 2 and value[0] == "__TOMBSTONE__"
        )

    def _start_background_compaction(self) -> None:
        """Start background compaction thread."""

        def background_worker():
            while True:
                time.sleep(60)  # Run every minute
                try:
                    self.compact_background()
                except Exception as e:
                    logger.error(f"Background compaction error: {e}")

        import threading

        thread = threading.Thread(target=background_worker, daemon=True)
        thread.start()

        logger.info("Background compaction thread started")

    async def get_async(self, key: str) -> Optional[Any]:
        """Async version of get."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.get, key)

    async def put_async(self, key: str, value: Any) -> None:
        """Async version of put."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.executor, self.put, key, value)

    def close(self) -> None:
        """Clean up resources."""
        # Flush remaining memtable
        if self.memtable:
            self.flush_memtable()

        # Shutdown executor
        self.executor.shutdown(wait=True)

        logger.info("MerkleLSM closed")
