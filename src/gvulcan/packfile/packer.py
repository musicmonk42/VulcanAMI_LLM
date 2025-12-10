"""
Pack File Writer and Reader Implementation

This module provides comprehensive packfile creation and reading with support for
compression, chunking, indexing, integrity verification, and bloom filters.
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import zstandard as zstd

from ..bloom import BloomFilter
from ..crc32c import CRC32CResult, crc32c, crc32c_detailed
from ..merkle import MerkleProof, MerkleTree, merkle_root
from .header import HEADER_SIZE, HeaderFlags, PackHeaderV2, create_header

logger = logging.getLogger(__name__)

# Constants
CHUNK_ALIGNMENT = 8192  # 8 KiB alignment for chunks
CRC32C_SIZE = 4  # Size of CRC32C checksum in bytes
INDEX_ENTRY_SIZE = 64  # Size of each index entry


class PackError(Exception):
    """Base exception for pack operations"""

    pass


class ChunkNotFoundError(PackError):
    """Raised when a chunk cannot be found in the pack"""

    pass


class IntegrityError(PackError):
    """Raised when integrity verification fails"""

    pass


class PackFullError(PackError):
    """Raised when pack cannot accept more chunks"""

    pass


@dataclass
class ChunkMetadata:
    """
    Metadata for a chunk in the packfile.

    Attributes:
        content_hash: Hash identifying the chunk
        offset: Byte offset in pack body
        size: Size of chunk data (excluding CRC)
        compressed_size: Size after compression
        crc32c: CRC32C checksum of chunk data
        flags: Chunk-specific flags
    """

    content_hash: bytes
    offset: int
    size: int
    compressed_size: int
    crc32c: int
    flags: int = 0

    def to_bytes(self) -> bytes:
        """Serialize metadata to 64-byte index entry"""
        # Format: 32s (hash) + Q (offset) + Q (size) + Q (compressed) + I (crc) + I (flags) + 4s (reserved)
        return struct.pack(
            "<32sQQQII4s",
            self.content_hash[:32],
            self.offset,
            self.size,
            self.compressed_size,
            self.crc32c,
            self.flags,
            b"\x00\x00\x00\x00",
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> ChunkMetadata:
        """Deserialize metadata from index entry"""
        if len(data) < INDEX_ENTRY_SIZE:
            raise ValueError(f"Index entry too short: {len(data)} bytes")

        (content_hash, offset, size, compressed_size, crc32c_val, flags, _) = (
            struct.unpack("<32sQQQII4s", data[:INDEX_ENTRY_SIZE])
        )

        return cls(
            content_hash=content_hash,
            offset=offset,
            size=size,
            compressed_size=compressed_size,
            crc32c=crc32c_val,
            flags=flags,
        )


@dataclass
class PackStats:
    """Statistics for a packfile"""

    chunk_count: int
    total_size: int
    compressed_size: int
    compression_ratio: float
    index_size: int
    header_size: int
    total_file_size: int


class PackWriter:
    """
    Write chunks to a packfile with compression, deduplication, and integrity checking.

    The packfile format:
    - Header (8192 bytes): Metadata, bloom filter, flags
    - Body: Compressed concatenation of [chunk_data + CRC32C] aligned to 8KiB
    - Index: Array of ChunkMetadata entries

    Example:
        writer = PackWriter(zstd_level=3)
        writer.add_chunk(b"chunk1", b"hash1")
        writer.add_chunk(b"chunk2", b"hash2")
        pack_bytes = writer.finalize()
    """

    def __init__(
        self,
        zstd_level: int = 3,
        bloom_size: int = 128,
        max_chunks: int = 100000,
        enable_deduplication: bool = True,
        chunk_alignment: int = CHUNK_ALIGNMENT,
    ):
        """
        Initialize pack writer.

        Args:
            zstd_level: Zstandard compression level (1-22)
            bloom_size: Size of bloom filter in bytes
            max_chunks: Maximum number of chunks allowed
            enable_deduplication: Whether to deduplicate chunks by hash
            chunk_alignment: Byte alignment for chunks
        """
        self.cctx = zstd.ZstdCompressor(level=zstd_level)
        self.header = PackHeaderV2()
        self.entries: List[ChunkMetadata] = []
        self.bloom = BloomFilter(size_bytes=bloom_size)
        self._body = bytearray()
        self._current_offset = 0
        self._hash_set: set[bytes] = set()  # For deduplication
        self._max_chunks = max_chunks
        self._enable_dedup = enable_deduplication
        self._chunk_alignment = chunk_alignment
        self._finalized = False

        logger.info(
            f"Initialized PackWriter: zstd_level={zstd_level}, "
            f"bloom_size={bloom_size}, max_chunks={max_chunks}"
        )

    def _align_offset(self, offset: int) -> int:
        """Align offset to chunk boundary"""
        remainder = offset % self._chunk_alignment
        if remainder == 0:
            return offset
        return offset + (self._chunk_alignment - remainder)

    def _add_padding(self) -> int:
        """Add padding to align to chunk boundary"""
        current_pos = len(self._body)
        aligned_pos = self._align_offset(current_pos)
        padding_size = aligned_pos - current_pos

        if padding_size > 0:
            self._body.extend(b"\x00" * padding_size)

        return padding_size

    def add_chunk(self, chunk_bytes: bytes, content_hash: bytes) -> bool:
        """
        Add a chunk to the packfile.

        Args:
            chunk_bytes: Raw chunk data
            content_hash: Hash identifying the chunk

        Returns:
            True if chunk was added, False if deduplicated

        Raises:
            PackFullError: If max_chunks limit reached
            PackError: If writer has been finalized
        """
        if self._finalized:
            raise PackError("Cannot add chunks after finalization")

        if len(self.entries) >= self._max_chunks:
            raise PackFullError(f"Pack full: {self._max_chunks} chunks")

        # Check for deduplication
        if self._enable_dedup and content_hash in self._hash_set:
            logger.debug(f"Chunk {content_hash.hex()[:16]} deduplicated")
            return False

        # Compute CRC32C
        crc_result = crc32c_detailed(chunk_bytes)
        crc_bytes = struct.pack("<I", crc_result.checksum)

        # Align to chunk boundary
        padding_size = self._add_padding()

        # Record current offset
        chunk_offset = len(self._body)

        # Append chunk data + CRC32C
        self._body.extend(chunk_bytes)
        self._body.extend(crc_bytes)

        # Create metadata
        metadata = ChunkMetadata(
            content_hash=content_hash,
            offset=chunk_offset,
            size=len(chunk_bytes),
            compressed_size=0,  # Will be set during finalization
            crc32c=crc_result.checksum,
            flags=0,
        )

        self.entries.append(metadata)
        self.bloom.add(content_hash)
        self._hash_set.add(content_hash)

        logger.debug(
            f"Added chunk {content_hash.hex()[:16]}: "
            f"size={len(chunk_bytes)}, offset={chunk_offset}, padding={padding_size}"
        )

        return True

    def add_chunks_batch(self, chunks: List[Tuple[bytes, bytes]]) -> Dict[str, int]:
        """
        Add multiple chunks in batch.

        Args:
            chunks: List of (chunk_bytes, content_hash) tuples

        Returns:
            Dictionary with statistics: added, deduplicated, failed
        """
        stats = {"added": 0, "deduplicated": 0, "failed": 0}

        for chunk_bytes, content_hash in chunks:
            try:
                if self.add_chunk(chunk_bytes, content_hash):
                    stats["added"] += 1
                else:
                    stats["deduplicated"] += 1
            except Exception as e:
                logger.error(f"Failed to add chunk: {e}")
                stats["failed"] += 1

        return stats

    def finalize(self) -> bytes:
        """
        Finalize the packfile and return complete binary data.

        Returns:
            Complete packfile bytes (header + compressed body + index)

        Raises:
            PackError: If already finalized
        """
        if self._finalized:
            raise PackError("Pack already finalized")

        logger.info(f"Finalizing pack with {len(self.entries)} chunks")

        # Compute Merkle root
        chunk_hashes = [e.content_hash for e in self.entries]
        merkle_root_hash = merkle_root(chunk_hashes) if chunk_hashes else bytes(32)

        # Compress body
        body_bytes = bytes(self._body)
        compressed_body = self.cctx.compress(body_bytes)

        logger.info(
            f"Compressed body: {len(body_bytes)} -> {len(compressed_body)} bytes "
            f"({100 * len(compressed_body) / max(len(body_bytes), 1):.1f}%)"
        )

        # Update compressed sizes in metadata
        compression_ratio = len(compressed_body) / max(len(body_bytes), 1)
        for entry in self.entries:
            entry.compressed_size = int(entry.size * compression_ratio)

        # Build index
        index_bytes = b"".join(entry.to_bytes() for entry in self.entries)

        # Calculate index offset (header + compressed body)
        index_offset = HEADER_SIZE + len(compressed_body)

        # Update header
        self.header.update_metadata(
            chunk_count=len(self.entries),
            index_offset=index_offset,
            merkle_root=merkle_root_hash,
            bloom_filter=self.bloom.to_bytes(),
        )

        # Set flags
        self.header.set_flag("compressed", True)
        self.header.set_flag("deduplicated", self._enable_dedup)
        self.header.set_flag("verified", True)

        # Pack everything together
        header_bytes = self.header.to_bytes()
        pack_bytes = header_bytes + compressed_body + index_bytes

        self._finalized = True

        logger.info(
            f"Pack finalized: total_size={len(pack_bytes)}, "
            f"header={len(header_bytes)}, body={len(compressed_body)}, "
            f"index={len(index_bytes)}"
        )

        return pack_bytes

    def get_stats(self) -> PackStats:
        """Get current pack statistics"""
        body_size = len(self._body)
        compressed_size = len(self.cctx.compress(bytes(self._body)))
        index_size = len(self.entries) * INDEX_ENTRY_SIZE

        return PackStats(
            chunk_count=len(self.entries),
            total_size=body_size,
            compressed_size=compressed_size,
            compression_ratio=compressed_size / max(body_size, 1),
            index_size=index_size,
            header_size=HEADER_SIZE,
            total_file_size=HEADER_SIZE + compressed_size + index_size,
        )


class PackReader:
    """
    Read chunks from a packfile with integrity verification and efficient lookups.

    Example:
        reader = PackReader(pack_bytes)
        chunk = reader.get_chunk(content_hash)
        if reader.verify_chunk(content_hash, chunk):
            print("Chunk verified!")
    """

    def __init__(self, blob: bytes, verify_on_init: bool = True):
        """
        Initialize pack reader.

        Args:
            blob: Complete packfile bytes
            verify_on_init: Whether to verify header on initialization

        Raises:
            PackError: If blob is too small or invalid
        """
        if len(blob) < HEADER_SIZE:
            raise PackError(f"Pack too small: {len(blob)} bytes")

        # Parse header
        self.blob = blob
        self.header = PackHeaderV2.from_bytes(blob[:HEADER_SIZE])

        if verify_on_init:
            logger.info(f"Loaded pack: {self.header.chunk_count} chunks")

        # Decompress body
        compressed_body = blob[HEADER_SIZE : self.header.index_offset]
        dctx = zstd.ZstdDecompressor()
        self.body = dctx.decompress(compressed_body)

        # Parse index
        index_start = self.header.index_offset
        index_end = index_start + (self.header.chunk_count * INDEX_ENTRY_SIZE)

        if index_end > len(blob):
            raise PackError("Index extends beyond pack size")

        index_bytes = blob[index_start:index_end]
        self.index: Dict[bytes, ChunkMetadata] = {}

        for i in range(self.header.chunk_count):
            entry_start = i * INDEX_ENTRY_SIZE
            entry_end = entry_start + INDEX_ENTRY_SIZE
            entry_bytes = index_bytes[entry_start:entry_end]

            metadata = ChunkMetadata.from_bytes(entry_bytes)
            self.index[metadata.content_hash] = metadata

        # Rebuild bloom filter
        self.bloom = BloomFilter(size_bytes=len(self.header.bloom_filter))
        self.bloom.from_bytes(self.header.bloom_filter)

        logger.info(f"Loaded {len(self.index)} chunks from pack")

    def has_chunk(self, content_hash: bytes) -> bool:
        """
        Quick check if chunk might exist (uses bloom filter).

        Args:
            content_hash: Hash to check

        Returns:
            True if chunk might exist, False if definitely does not
        """
        return self.bloom.contains(content_hash)

    def get_chunk(self, content_hash: bytes, verify: bool = True) -> Optional[bytes]:
        """
        Get chunk data by content hash.

        Args:
            content_hash: Hash identifying the chunk
            verify: Whether to verify CRC32C checksum

        Returns:
            Chunk bytes, or None if not found

        Raises:
            IntegrityError: If CRC32C verification fails
        """
        # Quick bloom filter check
        if not self.has_chunk(content_hash):
            return None

        # Lookup in index
        metadata = self.index.get(content_hash)
        if metadata is None:
            return None

        # Extract chunk data
        start = metadata.offset
        end = start + metadata.size

        if end > len(self.body):
            raise PackError(
                f"Chunk extends beyond body: offset={start}, size={metadata.size}"
            )

        chunk_data = self.body[start:end]

        # Verify CRC32C
        if verify:
            crc_start = end
            crc_end = crc_start + CRC32C_SIZE

            if crc_end > len(self.body):
                raise PackError("CRC32C missing")

            stored_crc = struct.unpack("<I", self.body[crc_start:crc_end])[0]
            computed_crc = crc32c(chunk_data)

            if stored_crc != computed_crc:
                raise IntegrityError(
                    f"CRC32C mismatch for chunk {content_hash.hex()[:16]}: "
                    f"expected={stored_crc:08x}, got={computed_crc:08x}"
                )

        return chunk_data

    def verify_chunk(self, content_hash: bytes, expected_data: bytes) -> bool:
        """
        Verify that stored chunk matches expected data.

        Args:
            content_hash: Hash identifying the chunk
            expected_data: Expected chunk data

        Returns:
            True if chunk matches
        """
        try:
            chunk_data = self.get_chunk(content_hash, verify=True)
            return chunk_data == expected_data
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return False

    def get_all_hashes(self) -> List[bytes]:
        """Get list of all chunk hashes in pack"""
        return list(self.index.keys())

    def iter_chunks(self, verify: bool = False) -> Iterator[Tuple[bytes, bytes]]:
        """
        Iterate over all chunks in pack.

        Args:
            verify: Whether to verify CRC32C for each chunk

        Yields:
            Tuples of (content_hash, chunk_data)
        """
        for content_hash, metadata in self.index.items():
            try:
                chunk_data = self.get_chunk(content_hash, verify=verify)
                if chunk_data is not None:
                    yield (content_hash, chunk_data)
            except Exception as e:
                logger.error(f"Error reading chunk {content_hash.hex()[:16]}: {e}")

    def get_metadata(self, content_hash: bytes) -> Optional[ChunkMetadata]:
        """Get metadata for a specific chunk"""
        return self.index.get(content_hash)

    def get_stats(self) -> PackStats:
        """Get pack statistics"""
        total_size = sum(m.size for m in self.index.values())
        compressed_size = len(self.blob[HEADER_SIZE : self.header.index_offset])
        index_size = len(self.index) * INDEX_ENTRY_SIZE

        return PackStats(
            chunk_count=len(self.index),
            total_size=total_size,
            compressed_size=compressed_size,
            compression_ratio=compressed_size / max(total_size, 1),
            index_size=index_size,
            header_size=HEADER_SIZE,
            total_file_size=len(self.blob),
        )

    def verify_merkle_root(self) -> bool:
        """
        Verify that stored Merkle root matches computed root.

        Returns:
            True if Merkle root is valid
        """
        chunk_hashes = list(self.index.keys())
        computed_root = merkle_root(chunk_hashes) if chunk_hashes else bytes(32)
        return computed_root == self.header.pack_merkle_root

    def __len__(self) -> int:
        """Return number of chunks in pack"""
        return len(self.index)

    def __contains__(self, content_hash: bytes) -> bool:
        """Check if chunk exists in pack"""
        return content_hash in self.index
