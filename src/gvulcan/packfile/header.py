"""
Pack File Header Implementation (GPK2 Format)

This module provides comprehensive binary header packing/unpacking for GPK2 packfiles
with support for metadata, integrity checks, bloom filters, and version management.
"""

from __future__ import annotations
import struct
from typing import Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Constants
MAGIC = b"GPK2"
VERSION = 2
HEADER_SIZE = 8192
BLOOM_SIZE = 128
INDEX_ENTRY_SIZE = 64
RESERVED_SIZE = 16

# Header format string (Little Endian):
# 4s  = magic (GPK2)
# H   = version (uint16)
# H   = flags (uint16)
# Q   = chunk_count (uint64)
# Q   = index_offset (uint64)
# 16s = reserved bytes
# 32s = pack_merkle_root
# 64s = packer_semver (padded)
# 128s = bloom_filter
# Remaining bytes = padding to 8192
HEADER_FMT = "<4sHHQQ16s32s64s128s"
HEADER_FIXED_SIZE = struct.calcsize(HEADER_FMT)


class PackHeaderError(Exception):
    """Base exception for pack header errors"""

    pass


class InvalidMagicError(PackHeaderError):
    """Raised when header magic bytes are invalid"""

    pass


class UnsupportedVersionError(PackHeaderError):
    """Raised when header version is not supported"""

    pass


class HeaderValidationError(PackHeaderError):
    """Raised when header validation fails"""

    pass


@dataclass
class HeaderFlags:
    """
    Bit flags for pack header configuration

    Bit layout:
    - 0: Compressed (1 if body is zstd compressed)
    - 1: Encrypted (1 if body is encrypted)
    - 2: Deduplicated (1 if chunks are deduplicated)
    - 3: Verified (1 if all chunks have been verified)
    - 4: Incremental (1 if this is an incremental pack)
    - 5-7: Reserved for future use
    - 8-15: Custom flags
    """

    compressed: bool = True
    encrypted: bool = False
    deduplicated: bool = True
    verified: bool = False
    incremental: bool = False
    custom: int = 0

    def to_int(self) -> int:
        """Convert flags to integer representation"""
        flags = 0
        if self.compressed:
            flags |= 1 << 0
        if self.encrypted:
            flags |= 1 << 1
        if self.deduplicated:
            flags |= 1 << 2
        if self.verified:
            flags |= 1 << 3
        if self.incremental:
            flags |= 1 << 4
        flags |= (self.custom & 0xFF) << 8
        return flags

    @classmethod
    def from_int(cls, flags: int) -> HeaderFlags:
        """Create flags from integer representation"""
        return cls(
            compressed=bool(flags & (1 << 0)),
            encrypted=bool(flags & (1 << 1)),
            deduplicated=bool(flags & (1 << 2)),
            verified=bool(flags & (1 << 3)),
            incremental=bool(flags & (1 << 4)),
            custom=(flags >> 8) & 0xFF,
        )

    def __str__(self) -> str:
        """Human-readable flags representation"""
        parts = []
        if self.compressed:
            parts.append("compressed")
        if self.encrypted:
            parts.append("encrypted")
        if self.deduplicated:
            parts.append("dedup")
        if self.verified:
            parts.append("verified")
        if self.incremental:
            parts.append("incremental")
        if self.custom:
            parts.append(f"custom=0x{self.custom:02x}")
        return ",".join(parts) if parts else "none"


class PackHeaderV2:
    """
    GPK2 packfile header with comprehensive metadata and integrity support.

    The header is always HEADER_SIZE bytes (8192) to enable direct memory mapping
    and efficient parsing without requiring variable-length header reads.

    Attributes:
        magic: Magic bytes identifying GPK2 format
        version: Format version (currently 2)
        flags: Configuration flags (HeaderFlags)
        chunk_count: Number of chunks in the pack
        index_offset: Byte offset to the chunk index
        reserved: Reserved bytes for future extensions
        pack_merkle_root: Merkle root of all chunk hashes
        packer_semver: Semantic version of packer tool
        bloom_filter: Bloom filter for chunk membership testing
    """

    def __init__(
        self,
        magic: bytes = MAGIC,
        version: int = VERSION,
        flags: Optional[HeaderFlags] = None,
        chunk_count: int = 0,
        index_offset: int = 0,
        reserved: bytes = None,
        pack_merkle_root: bytes = None,
        packer_semver: bytes = None,
        bloom_filter: bytes = None,
    ):
        self.magic = magic
        self.version = version
        self.flags = flags or HeaderFlags()
        self.chunk_count = chunk_count
        self.index_offset = index_offset
        self.reserved = reserved or bytes(RESERVED_SIZE)
        self.pack_merkle_root = pack_merkle_root or bytes(32)
        self.packer_semver = packer_semver or b"460"
        self.bloom_filter = bloom_filter or bytes(BLOOM_SIZE)

        # Validate initialization
        self._validate()

    def _validate(self) -> None:
        """Validate header fields"""
        if self.magic != MAGIC:
            raise InvalidMagicError(f"Invalid magic bytes: {self.magic!r}")

        if self.version != VERSION:
            raise UnsupportedVersionError(f"Unsupported version: {self.version}")

        if len(self.reserved) != RESERVED_SIZE:
            raise HeaderValidationError(f"Reserved bytes must be {RESERVED_SIZE} bytes")

        if len(self.pack_merkle_root) != 32:
            raise HeaderValidationError("Merkle root must be 32 bytes")

        if len(self.bloom_filter) != BLOOM_SIZE:
            raise HeaderValidationError(f"Bloom filter must be {BLOOM_SIZE} bytes")

        if self.chunk_count < 0:
            raise HeaderValidationError("Chunk count cannot be negative")

        if self.index_offset < 0:
            raise HeaderValidationError("Index offset cannot be negative")

    def to_bytes(self) -> bytes:
        """
        Pack header to binary format (8192 bytes).

        Returns:
            8192-byte packed header

        Raises:
            HeaderValidationError: If header fields are invalid
        """
        self._validate()

        # Pad/truncate packer_semver to 64 bytes
        semver_padded = self.packer_semver[:64].ljust(64, b"\x00")

        # Pack fixed header fields
        fixed_data = struct.pack(
            HEADER_FMT,
            self.magic,
            self.version,
            self.flags.to_int(),
            self.chunk_count,
            self.index_offset,
            self.reserved,
            self.pack_merkle_root,
            semver_padded,
            self.bloom_filter,
        )

        # Pad to HEADER_SIZE with zeros
        padding_size = HEADER_SIZE - len(fixed_data)
        if padding_size < 0:
            raise HeaderValidationError(
                f"Fixed header size {len(fixed_data)} exceeds HEADER_SIZE {HEADER_SIZE}"
            )

        return fixed_data + (b"\x00" * padding_size)

    @classmethod
    def from_bytes(cls, data: bytes) -> PackHeaderV2:
        """
        Unpack header from binary format.

        Args:
            data: Binary header data (must be at least HEADER_FIXED_SIZE bytes)

        Returns:
            Unpacked PackHeaderV2 instance

        Raises:
            HeaderValidationError: If data is too short or invalid
            InvalidMagicError: If magic bytes don't match
            UnsupportedVersionError: If version is not supported
        """
        if len(data) < HEADER_FIXED_SIZE:
            raise HeaderValidationError(
                f"Header data too short: {len(data)} bytes, need at least {HEADER_FIXED_SIZE}"
            )

        # Unpack fixed fields
        (
            magic,
            version,
            flags_int,
            chunk_count,
            index_offset,
            reserved,
            pack_merkle_root,
            packer_semver_padded,
            bloom_filter,
        ) = struct.unpack(HEADER_FMT, data[:HEADER_FIXED_SIZE])

        # Strip null padding from semver
        packer_semver = packer_semver_padded.rstrip(b"\x00")

        # Parse flags
        flags = HeaderFlags.from_int(flags_int)

        # Create header instance (validation happens in __init__)
        return cls(
            magic=magic,
            version=version,
            flags=flags,
            chunk_count=chunk_count,
            index_offset=index_offset,
            reserved=reserved,
            pack_merkle_root=pack_merkle_root,
            packer_semver=packer_semver,
            bloom_filter=bloom_filter,
        )

    def update_metadata(
        self,
        chunk_count: Optional[int] = None,
        index_offset: Optional[int] = None,
        merkle_root: Optional[bytes] = None,
        bloom_filter: Optional[bytes] = None,
    ) -> None:
        """
        Update header metadata fields.

        Args:
            chunk_count: New chunk count
            index_offset: New index offset
            merkle_root: New Merkle root hash
            bloom_filter: New bloom filter bytes
        """
        if chunk_count is not None:
            self.chunk_count = chunk_count
        if index_offset is not None:
            self.index_offset = index_offset
        if merkle_root is not None:
            if len(merkle_root) != 32:
                raise HeaderValidationError("Merkle root must be 32 bytes")
            self.pack_merkle_root = merkle_root
        if bloom_filter is not None:
            if len(bloom_filter) != BLOOM_SIZE:
                raise HeaderValidationError(f"Bloom filter must be {BLOOM_SIZE} bytes")
            self.bloom_filter = bloom_filter

        self._validate()

    def set_flag(self, flag_name: str, value: bool) -> None:
        """
        Set a specific flag by name.

        Args:
            flag_name: Name of flag (compressed, encrypted, etc.)
            value: New flag value
        """
        if hasattr(self.flags, flag_name):
            setattr(self.flags, flag_name, value)
        else:
            raise ValueError(f"Unknown flag: {flag_name}")

    def get_info(self) -> dict:
        """
        Get human-readable header information.

        Returns:
            Dictionary with header metadata
        """
        return {
            "magic": self.magic.decode("ascii", errors="replace"),
            "version": self.version,
            "flags": str(self.flags),
            "chunk_count": self.chunk_count,
            "index_offset": self.index_offset,
            "merkle_root": self.pack_merkle_root.hex()
            if self.pack_merkle_root != bytes(32)
            else None,
            "packer_version": self.packer_semver.decode(
                "ascii", errors="replace"
            ).rstrip("\x00"),
            "bloom_filter_size": len(self.bloom_filter),
            "header_size": HEADER_SIZE,
        }

    def __repr__(self) -> str:
        """String representation of header"""
        return (
            f"PackHeaderV2(magic={self.magic!r}, version={self.version}, "
            f"flags={self.flags}, chunk_count={self.chunk_count}, "
            f"index_offset={self.index_offset})"
        )

    def __str__(self) -> str:
        """Human-readable string representation"""
        info = self.get_info()
        return (
            f"GPK2 Header v{info['version']}\n"
            f"  Chunks: {info['chunk_count']}\n"
            f"  Flags: {info['flags']}\n"
            f"  Index offset: {info['index_offset']}\n"
            f"  Packer: {info['packer_version']}"
        )


def create_header(
    chunk_count: int,
    index_offset: int,
    merkle_root: bytes,
    bloom_filter: bytes,
    compressed: bool = True,
    deduplicated: bool = True,
    packer_version: str = "460",
) -> PackHeaderV2:
    """
    Convenience function to create a pack header.

    Args:
        chunk_count: Number of chunks in pack
        index_offset: Byte offset to chunk index
        merkle_root: Merkle root of all chunks
        bloom_filter: Bloom filter bytes
        compressed: Whether pack body is compressed
        deduplicated: Whether chunks are deduplicated
        packer_version: Version string of packer tool

    Returns:
        Configured PackHeaderV2 instance
    """
    flags = HeaderFlags(compressed=compressed, deduplicated=deduplicated)

    return PackHeaderV2(
        flags=flags,
        chunk_count=chunk_count,
        index_offset=index_offset,
        pack_merkle_root=merkle_root,
        bloom_filter=bloom_filter,
        packer_semver=packer_version.encode("ascii"),
    )


def validate_header_bytes(data: bytes) -> bool:
    """
    Quick validation of header bytes without full parsing.

    Args:
        data: Header bytes to validate

    Returns:
        True if header appears valid
    """
    try:
        if len(data) < 6:
            return False

        magic = data[:4]
        if magic != MAGIC:
            return False

        version = struct.unpack("<H", data[4:6])[0]
        if version != VERSION:
            return False

        return True
    except Exception:
        return False
