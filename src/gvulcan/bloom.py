"""
Bloom Filter Implementation

This module provides a comprehensive Bloom filter implementation with support for
set operations, persistence, statistics, and optimized false positive rates.
"""

from __future__ import annotations
import hashlib
import math
import json
from typing import Iterable, Optional, List, Set, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class BloomStats:
    """
    Statistics for a Bloom filter

    Attributes:
        size_bytes: Size of the filter in bytes
        size_bits: Size of the filter in bits
        k: Number of hash functions
        items_added: Estimated number of items added
        fill_ratio: Ratio of set bits to total bits
        estimated_fpp: Estimated false positive probability
    """

    size_bytes: int
    size_bits: int
    k: int
    items_added: int
    fill_ratio: float
    estimated_fpp: float

    def to_dict(self) -> dict:
        """Convert stats to dictionary"""
        return {
            "size_bytes": self.size_bytes,
            "size_bits": self.size_bits,
            "k": self.k,
            "items_added": self.items_added,
            "fill_ratio": f"{self.fill_ratio:.4f}",
            "estimated_fpp": f"{self.estimated_fpp:.6f}",
        }


class BloomFilter:
    """
    Space-efficient probabilistic data structure for set membership testing.

    A Bloom filter allows you to test whether an element is a member of a set with:
    - No false negatives (if it says "not in set", it's definitely not)
    - Possible false positives (if it says "in set", it might not be)

    Features:
    - Configurable false positive rate
    - Set operations (union, intersection)
    - Persistence (save/load)
    - Statistics and monitoring
    - Multiple hash function strategies
    """

    def __init__(self, size_bytes: int = 128, k: int = 7):
        """
        Initialize a Bloom filter.

        Args:
            size_bytes: Size of the bit array in bytes
            k: Number of hash functions to use
        """
        self.size = size_bytes
        self.bits = bytearray(self.size)
        self.k = k
        self.m = self.size * 8  # Total number of bits
        self.items_added = 0

        logger.info(
            f"Initialized BloomFilter: {self.size} bytes, {self.k} hash functions"
        )

    @classmethod
    def create_optimal(
        cls, expected_items: int, false_positive_rate: float = 0.01
    ) -> BloomFilter:
        """
        Create a Bloom filter with optimal parameters for given constraints.

        Args:
            expected_items: Expected number of items to add
            false_positive_rate: Desired false positive probability

        Returns:
            BloomFilter with optimal size and hash function count

        Example:
            >>> bf = BloomFilter.create_optimal(10000, 0.001)
        """
        # Calculate optimal bit array size: m = -n*ln(p) / (ln(2)^2)
        m = -expected_items * math.log(false_positive_rate) / (math.log(2) ** 2)
        m = int(math.ceil(m))

        # Calculate optimal number of hash functions: k = (m/n) * ln(2)
        k = (m / expected_items) * math.log(2)
        k = int(math.ceil(k))

        size_bytes = (m + 7) // 8  # Round up to nearest byte

        logger.info(
            f"Created optimal BloomFilter for {expected_items} items "
            f"with FPR {false_positive_rate}: {size_bytes} bytes, {k} hashes"
        )

        return cls(size_bytes=size_bytes, k=k)

    def _hashes(self, item: bytes) -> List[int]:
        """
        Generate k hash values for an item using double hashing.

        Args:
            item: Item to hash

        Returns:
            List of k bit positions
        """
        h = hashlib.sha256(item).digest()
        h1 = int.from_bytes(h[:16], "big")
        h2 = int.from_bytes(h[16:], "big")
        return [(h1 + i * h2) % self.m for i in range(self.k)]

    def add(self, item: bytes) -> None:
        """
        Add an item to the Bloom filter.

        Args:
            item: Item to add (as bytes)
        """
        for idx in self._hashes(item):
            self.bits[idx // 8] |= 1 << (idx % 8)
        self.items_added += 1
        logger.debug(f"Added item to BloomFilter (total: {self.items_added})")

    def add_str(self, item: str) -> None:
        """
        Add a string item to the Bloom filter.

        Args:
            item: String to add
        """
        self.add(item.encode("utf-8"))

    def add_many(self, items: Iterable[bytes]) -> int:
        """
        Add multiple items to the Bloom filter.

        Args:
            items: Iterable of items to add

        Returns:
            Number of items added
        """
        count = 0
        for item in items:
            self.add(item)
            count += 1

        logger.info(f"Added {count} items to BloomFilter")
        return count

    def might_contain(self, item: bytes) -> bool:
        """
        Test if an item might be in the set.

        Args:
            item: Item to test

        Returns:
            False if definitely not in set, True if possibly in set
        """
        for idx in self._hashes(item):
            if not (self.bits[idx // 8] & (1 << (idx % 8))):
                return False
        return True

    def might_contain_str(self, item: str) -> bool:
        """
        Test if a string item might be in the set.

        Args:
            item: String to test

        Returns:
            False if definitely not in set, True if possibly in set
        """
        return self.might_contain(item.encode("utf-8"))

    def __contains__(self, item: bytes) -> bool:
        """Support 'in' operator"""
        return self.might_contain(item)

    def union(self, other: BloomFilter) -> BloomFilter:
        """
        Compute the union of two Bloom filters.

        Args:
            other: Another BloomFilter

        Returns:
            New BloomFilter containing union

        Raises:
            ValueError: If filters have incompatible parameters
        """
        if self.size != other.size or self.k != other.k:
            raise ValueError("Bloom filters must have same size and k for union")

        result = BloomFilter(size_bytes=self.size, k=self.k)
        for i in range(self.size):
            result.bits[i] = self.bits[i] | other.bits[i]

        result.items_added = self.items_added + other.items_added
        logger.info(f"Computed union of two BloomFilters")

        return result

    def intersection(self, other: BloomFilter) -> BloomFilter:
        """
        Compute the intersection of two Bloom filters (approximate).

        Note: Intersection of Bloom filters is approximate and may have
        higher false positive rates.

        Args:
            other: Another BloomFilter

        Returns:
            New BloomFilter containing intersection

        Raises:
            ValueError: If filters have incompatible parameters
        """
        if self.size != other.size or self.k != other.k:
            raise ValueError("Bloom filters must have same size and k for intersection")

        result = BloomFilter(size_bytes=self.size, k=self.k)
        for i in range(self.size):
            result.bits[i] = self.bits[i] & other.bits[i]

        # Estimate items in intersection (rough approximation)
        result.items_added = min(self.items_added, other.items_added)
        logger.info(f"Computed intersection of two BloomFilters")

        return result

    def clear(self) -> None:
        """Clear all items from the filter"""
        self.bits = bytearray(self.size)
        self.items_added = 0
        logger.info("Cleared BloomFilter")

    def get_fill_ratio(self) -> float:
        """
        Get the ratio of set bits to total bits.

        Returns:
            Fill ratio between 0 and 1
        """
        set_bits = sum(bin(byte).count("1") for byte in self.bits)
        return set_bits / self.m

    def estimate_false_positive_probability(self) -> float:
        """
        Estimate the current false positive probability.

        Returns:
            Estimated FPP based on current fill ratio
        """
        fill_ratio = self.get_fill_ratio()
        # FPP ≈ (1 - e^(-kn/m))^k ≈ fill_ratio^k
        fpp = fill_ratio**self.k
        return fpp

    def get_stats(self) -> BloomStats:
        """
        Get comprehensive statistics about the filter.

        Returns:
            BloomStats object
        """
        return BloomStats(
            size_bytes=self.size,
            size_bits=self.m,
            k=self.k,
            items_added=self.items_added,
            fill_ratio=self.get_fill_ratio(),
            estimated_fpp=self.estimate_false_positive_probability(),
        )

    def to_bytes(self) -> bytes:
        """
        Serialize the filter to bytes.

        Returns:
            Byte representation of the filter
        """
        return bytes(self.bits)

    @classmethod
    def from_bytes(cls, data: bytes, k: int = 7) -> BloomFilter:
        """
        Deserialize a filter from bytes.

        Args:
            data: Byte data
            k: Number of hash functions used

        Returns:
            Restored BloomFilter
        """
        bf = cls(size_bytes=len(data), k=k)
        bf.bits[:] = data
        logger.info(f"Loaded BloomFilter from bytes: {len(data)} bytes, k={k}")
        return bf

    def save(self, path: Path) -> None:
        """
        Save the filter to a file.

        Args:
            path: Path to save to
        """
        metadata = {
            "size": self.size,
            "k": self.k,
            "items_added": self.items_added,
            "data": self.to_bytes().hex(),
        }

        with open(path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved BloomFilter to {path}")

    @classmethod
    def load(cls, path: Path) -> BloomFilter:
        """
        Load a filter from a file.

        Args:
            path: Path to load from

        Returns:
            Loaded BloomFilter
        """
        with open(path, "r") as f:
            metadata = json.load(f)

        data = bytes.fromhex(metadata["data"])
        bf = cls.from_bytes(data, k=metadata["k"])
        bf.items_added = metadata["items_added"]

        logger.info(f"Loaded BloomFilter from {path}")
        return bf

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"BloomFilter(size={self.size}B, k={self.k}, "
            f"items={self.items_added}, fill={stats.fill_ratio:.2%}, "
            f"fpp≈{stats.estimated_fpp:.4f})"
        )


class CountingBloomFilter:
    """
    Bloom filter that supports deletion by maintaining counters instead of bits.

    This variant uses counters instead of single bits, allowing items to be
    removed at the cost of increased memory usage.
    """

    def __init__(self, size_bytes: int = 128, k: int = 7, counter_bits: int = 4):
        """
        Initialize a Counting Bloom filter.

        Args:
            size_bytes: Base size in bytes
            k: Number of hash functions
            counter_bits: Bits per counter (max count = 2^counter_bits - 1)
        """
        self.k = k
        self.counter_bits = counter_bits
        self.max_count = (1 << counter_bits) - 1

        # Calculate number of counters
        self.num_counters = (size_bytes * 8) // counter_bits
        self.m = self.num_counters

        # Store counters as list of integers
        self.counters = [0] * self.num_counters
        self.items_added = 0

        logger.info(
            f"Initialized CountingBloomFilter: "
            f"{self.num_counters} counters, {counter_bits} bits each"
        )

    def _hashes(self, item: bytes) -> List[int]:
        """Generate k hash values for an item"""
        h = hashlib.sha256(item).digest()
        h1 = int.from_bytes(h[:16], "big")
        h2 = int.from_bytes(h[16:], "big")
        return [(h1 + i * h2) % self.m for i in range(self.k)]

    def add(self, item: bytes) -> None:
        """Add an item, incrementing counters"""
        for idx in self._hashes(item):
            if self.counters[idx] < self.max_count:
                self.counters[idx] += 1
        self.items_added += 1

    def remove(self, item: bytes) -> None:
        """
        Remove an item, decrementing counters.

        Args:
            item: Item to remove
        """
        for idx in self._hashes(item):
            if self.counters[idx] > 0:
                self.counters[idx] -= 1
        self.items_added = max(0, self.items_added - 1)

    def might_contain(self, item: bytes) -> bool:
        """Test if item might be in the set"""
        for idx in self._hashes(item):
            if self.counters[idx] == 0:
                return False
        return True

    def clear(self) -> None:
        """Clear all counters"""
        self.counters = [0] * self.num_counters
        self.items_added = 0


class ScalableBloomFilter:
    """
    Bloom filter that grows as more items are added.

    This implementation maintains multiple Bloom filters and adds new ones
    as the false positive rate increases, providing better performance for
    dynamic-sized sets.
    """

    def __init__(
        self,
        initial_capacity: int = 1000,
        error_rate: float = 0.001,
        growth_factor: int = 2,
    ):
        """
        Initialize a Scalable Bloom filter.

        Args:
            initial_capacity: Initial capacity
            error_rate: Target error rate
            growth_factor: Factor to grow capacity by
        """
        self.initial_capacity = initial_capacity
        self.error_rate = error_rate
        self.growth_factor = growth_factor
        self.filters: List[BloomFilter] = []
        self.items_added = 0

        # Create first filter
        self._add_filter(initial_capacity)

    def _add_filter(self, capacity: int) -> None:
        """Add a new filter with given capacity"""
        # Tighten error rate for each successive filter
        error_rate = self.error_rate * (0.5 ** len(self.filters))
        bf = BloomFilter.create_optimal(capacity, error_rate)
        self.filters.append(bf)
        logger.info(f"Added filter #{len(self.filters)} with capacity {capacity}")

    def add(self, item: bytes) -> None:
        """Add an item, creating new filter if needed"""
        # Check if current filter is getting full
        current = self.filters[-1]
        if current.get_fill_ratio() > 0.5:
            new_capacity = self.initial_capacity * (
                self.growth_factor ** len(self.filters)
            )
            self._add_filter(new_capacity)

        self.filters[-1].add(item)
        self.items_added += 1

    def might_contain(self, item: bytes) -> bool:
        """Test if item might be in any of the filters"""
        return any(bf.might_contain(item) for bf in self.filters)

    def get_stats(self) -> dict:
        """Get statistics about all filters"""
        return {
            "num_filters": len(self.filters),
            "total_items": self.items_added,
            "filters": [bf.get_stats().to_dict() for bf in self.filters],
        }


def test_false_positive_rate(
    bf: BloomFilter, test_items: Set[bytes], non_member_items: Set[bytes]
) -> Tuple[float, int, int]:
    """
    Test the actual false positive rate of a Bloom filter.

    Args:
        bf: BloomFilter to test
        test_items: Items that were added to the filter
        non_member_items: Items that were NOT added

    Returns:
        Tuple of (false_positive_rate, false_positives, tests)
    """
    # Verify no false negatives
    for item in test_items:
        if not bf.might_contain(item):
            logger.error("FALSE NEGATIVE DETECTED!")

    # Count false positives
    false_positives = sum(1 for item in non_member_items if bf.might_contain(item))

    fpr = false_positives / len(non_member_items) if non_member_items else 0.0

    logger.info(f"FPR Test: {false_positives}/{len(non_member_items)} = {fpr:.4%}")

    return fpr, false_positives, len(non_member_items)


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)

    print("=== Testing Basic Bloom Filter ===")
    bf = BloomFilter.create_optimal(expected_items=1000, false_positive_rate=0.01)

    # Add some items
    items = [f"item{i}".encode() for i in range(100)]
    bf.add_many(items)

    # Test membership
    print(f"'item50' in filter: {bf.might_contain(b'item50')}")
    print(f"'item999' in filter: {bf.might_contain(b'item999')}")

    # Get statistics
    stats = bf.get_stats()
    print(f"\nFilter stats: {stats.to_dict()}")

    print("\n=== Testing Set Operations ===")
    bf1 = BloomFilter(size_bytes=128, k=7)
    bf2 = BloomFilter(size_bytes=128, k=7)

    bf1.add_many([b"a", b"b", b"c"])
    bf2.add_many([b"b", b"c", b"d"])

    union = bf1.union(bf2)
    print(f"Union might contain 'a': {union.might_contain(b'a')}")
    print(f"Union might contain 'd': {union.might_contain(b'd')}")

    print("\n=== Testing Scalable Bloom Filter ===")
    sbf = ScalableBloomFilter(initial_capacity=100, error_rate=0.01)

    # Add many items to trigger scaling
    for i in range(500):
        sbf.add(f"item{i}".encode())

    print(f"Scalable filter stats: {sbf.get_stats()}")
