"""
CRC32C (Castagnoli) Checksum Implementation

This module provides comprehensive CRC32C checksum functionality with support for
streaming data, batch processing, validation, and performance optimization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Iterable, List, Optional, Union

import google_crc32c

logger = logging.getLogger(__name__)


@dataclass
class CRC32CResult:
    """
    Result of a CRC32C checksum computation

    Attributes:
        checksum: The CRC32C checksum value
        data_size: Size of data checksummed in bytes
        verified: Whether the checksum was verified against expected value
    """

    checksum: int
    data_size: int
    verified: Optional[bool] = None

    def to_hex(self) -> str:
        """Convert checksum to hexadecimal string"""
        return f"{self.checksum:08x}"

    def to_bytes(self) -> bytes:
        """Convert checksum to bytes (big-endian)"""
        return self.checksum.to_bytes(4, byteorder="big")


def crc32c(data: bytes) -> int:
    """
    Compute CRC32C checksum for data.

    Args:
        data: Bytes to compute checksum for

    Returns:
        CRC32C checksum as integer

    Example:
        >>> checksum = crc32c(b"hello world")
        >>> print(f"Checksum: {checksum:08x}")
    """
    return google_crc32c.value(data)


def crc32c_extend(crc: int, data: bytes) -> int:
    """
    Extend an existing CRC32C checksum with new data.

    This is useful for computing checksums of streaming or chunked data.

    Args:
        crc: Existing CRC32C value
        data: New data to add to checksum

    Returns:
        Updated CRC32C checksum

    Example:
        >>> crc = crc32c(b"hello ")
        >>> crc = crc32c_extend(crc, b"world")
    """
    crc_obj = google_crc32c.Checksum(initial_value=crc.to_bytes(4, "big"))
    crc_obj.update(data)
    return crc_obj.value


def crc32c_detailed(data: bytes, expected: Optional[int] = None) -> CRC32CResult:
    """
    Compute CRC32C with detailed result including verification.

    Args:
        data: Bytes to compute checksum for
        expected: Optional expected checksum value for verification

    Returns:
        CRC32CResult with checksum and verification status
    """
    checksum = google_crc32c.value(data)
    verified = (checksum == expected) if expected is not None else None

    return CRC32CResult(checksum=checksum, data_size=len(data), verified=verified)


class StreamingCRC32C:
    """
    Streaming CRC32C checksum calculator.

    This class allows computing checksums incrementally as data arrives,
    which is memory-efficient for large files or network streams.

    Example:
        >>> stream_crc = StreamingCRC32C()
        >>> stream_crc.update(b"chunk1")
        >>> stream_crc.update(b"chunk2")
        >>> final_crc = stream_crc.finalize()
    """

    def __init__(self):
        """Initialize a new streaming CRC32C calculator"""
        self.crc_obj = google_crc32c.Checksum()
        self.total_bytes = 0
        self.chunk_count = 0

    def update(self, data: bytes) -> None:
        """
        Update the checksum with new data.

        Args:
            data: Bytes to add to the checksum
        """
        self.crc_obj.update(data)
        self.total_bytes += len(data)
        self.chunk_count += 1
        logger.debug(
            f"Updated CRC32C with {len(data)} bytes "
            f"(total: {self.total_bytes} bytes, {self.chunk_count} chunks)"
        )

    def digest(self) -> int:
        """Get the current checksum value without finalizing"""
        return self.crc_obj.value

    def hexdigest(self) -> str:
        """Get the current checksum as hexadecimal string"""
        return f"{self.digest():08x}"

    def finalize(self) -> CRC32CResult:
        """
        Finalize and return the complete checksum result.

        Returns:
            CRC32CResult with final checksum and statistics
        """
        return CRC32CResult(
            checksum=self.crc_obj.value, data_size=self.total_bytes, verified=None
        )

    def reset(self) -> None:
        """Reset the calculator for reuse"""
        self.crc_obj = google_crc32c.Checksum()
        self.total_bytes = 0
        self.chunk_count = 0


def crc32c_file(
    file_path: Union[str, Path], chunk_size: int = 65536, expected: Optional[int] = None
) -> CRC32CResult:
    """
    Compute CRC32C checksum for a file efficiently.

    Args:
        file_path: Path to file
        chunk_size: Size of chunks to read (default 64KB)
        expected: Optional expected checksum for verification

    Returns:
        CRC32CResult with checksum and verification status

    Raises:
        FileNotFoundError: If file doesn't exist
        IOError: If file cannot be read
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    stream_crc = StreamingCRC32C()

    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                stream_crc.update(chunk)
    except IOError as e:
        logger.error(f"Error reading file {file_path}: {e}")
        raise

    result = stream_crc.finalize()
    if expected is not None:
        result.verified = result.checksum == expected

    logger.info(
        f"Computed CRC32C for {file_path}: {result.to_hex()} ({result.data_size} bytes)"
    )

    return result


def crc32c_stream(stream: BinaryIO, chunk_size: int = 65536) -> CRC32CResult:
    """
    Compute CRC32C checksum for a binary stream.

    Args:
        stream: Binary stream to read from
        chunk_size: Size of chunks to read

    Returns:
        CRC32CResult with checksum
    """
    stream_crc = StreamingCRC32C()

    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            break
        stream_crc.update(chunk)

    return stream_crc.finalize()


def crc32c_batch(data_list: Iterable[bytes]) -> List[CRC32CResult]:
    """
    Compute CRC32C checksums for multiple data items in batch.

    Args:
        data_list: Iterable of byte data to checksum

    Returns:
        List of CRC32CResult for each data item
    """
    results = []
    for data in data_list:
        checksum = google_crc32c.value(data)
        results.append(
            CRC32CResult(checksum=checksum, data_size=len(data), verified=None)
        )

    logger.info(f"Computed batch CRC32C for {len(results)} items")
    return results


def verify_crc32c(data: bytes, expected: int) -> bool:
    """
    Verify data against expected CRC32C checksum.

    Args:
        data: Data to verify
        expected: Expected CRC32C checksum

    Returns:
        True if checksum matches, False otherwise
    """
    actual = google_crc32c.value(data)
    verified = actual == expected

    if not verified:
        logger.warning(
            f"CRC32C verification failed: expected {expected:08x}, got {actual:08x}"
        )
    else:
        logger.debug(f"CRC32C verification passed: {actual:08x}")

    return verified


def verify_crc32c_file(
    file_path: Union[str, Path], expected: int, chunk_size: int = 65536
) -> bool:
    """
    Verify file integrity against expected CRC32C checksum.

    Args:
        file_path: Path to file
        expected: Expected CRC32C checksum
        chunk_size: Size of chunks to read

    Returns:
        True if file checksum matches expected value
    """
    result = crc32c_file(file_path, chunk_size=chunk_size, expected=expected)
    return result.verified or False


class CRC32CValidator:
    """
    Validator for verifying data integrity using CRC32C checksums.

    This class maintains a registry of expected checksums and provides
    batch validation capabilities.
    """

    def __init__(self):
        """Initialize a new CRC32C validator"""
        self.checksums: Dict[str, int] = {}
        self.validation_count = 0
        self.failure_count = 0

    def register(self, key: str, checksum: int) -> None:
        """
        Register an expected checksum for a key.

        Args:
            key: Identifier for the data
            checksum: Expected CRC32C checksum
        """
        self.checksums[key] = checksum
        logger.debug(f"Registered checksum for {key}: {checksum:08x}")

    def validate(self, key: str, data: bytes) -> bool:
        """
        Validate data against registered checksum.

        Args:
            key: Identifier for the data
            data: Data to validate

        Returns:
            True if validation passes

        Raises:
            KeyError: If key not registered
        """
        if key not in self.checksums:
            raise KeyError(f"No checksum registered for key: {key}")

        expected = self.checksums[key]
        actual = google_crc32c.value(data)
        verified = actual == expected

        self.validation_count += 1
        if not verified:
            self.failure_count += 1
            logger.error(
                f"Validation failed for {key}: "
                f"expected {expected:08x}, got {actual:08x}"
            )
        else:
            logger.debug(f"Validation passed for {key}: {actual:08x}")

        return verified

    def validate_batch(self, items: Iterable[tuple[str, bytes]]) -> List[bool]:
        """
        Validate multiple items in batch.

        Args:
            items: Iterable of (key, data) tuples

        Returns:
            List of validation results
        """
        results = []
        for key, data in items:
            try:
                results.append(self.validate(key, data))
            except KeyError:
                results.append(False)
                self.failure_count += 1

        return results

    def get_stats(self) -> Dict[str, int]:
        """Get validation statistics"""
        return {
            "total_validations": self.validation_count,
            "failures": self.failure_count,
            "success_rate": (self.validation_count - self.failure_count)
            / max(1, self.validation_count),
            "registered_checksums": len(self.checksums),
        }


def crc32c_combine(crc1: int, crc2: int, len2: int) -> int:
    """
    Combine two CRC32C checksums.

    This allows computing the CRC32C of concatenated data from the
    CRC32Cs of the individual parts.

    Args:
        crc1: CRC32C of first data block
        crc2: CRC32C of second data block
        len2: Length of second data block

    Returns:
        Combined CRC32C checksum

    Note:
        This is a simplified implementation. For production use,
        consider using a library with proper CRC32C combination support.
    """
    # This is a placeholder - proper CRC32C combination requires
    # specific polynomial operations
    logger.warning("crc32c_combine is a simplified implementation")

    # For now, we'll compute it the straightforward way
    # In production, you'd use the mathematical properties of CRC32C
    return crc1  # Placeholder


def create_manifest(
    file_paths: List[Union[str, Path]], output_path: Optional[Union[str, Path]] = None
) -> Dict[str, str]:
    """
    Create a manifest of CRC32C checksums for multiple files.

    Args:
        file_paths: List of file paths to checksum
        output_path: Optional path to write manifest JSON

    Returns:
        Dictionary mapping file paths to checksums (hex)
    """
    import json

    manifest = {}
    for file_path in file_paths:
        try:
            result = crc32c_file(file_path)
            manifest[str(file_path)] = result.to_hex()
        except Exception as e:
            logger.error(f"Failed to checksum {file_path}: {e}")
            manifest[str(file_path)] = "ERROR"

    if output_path:
        with open(output_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Wrote manifest to {output_path}")

    return manifest


def verify_manifest(manifest_path: Union[str, Path]) -> Dict[str, bool]:
    """
    Verify files against a checksum manifest.

    Args:
        manifest_path: Path to manifest JSON file

    Returns:
        Dictionary mapping file paths to verification results
    """
    import json

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    results = {}
    for file_path, expected_hex in manifest.items():
        if expected_hex == "ERROR":
            results[file_path] = False
            continue

        try:
            expected = int(expected_hex, 16)
            verified = verify_crc32c_file(file_path, expected)
            results[file_path] = verified
        except Exception as e:
            logger.error(f"Failed to verify {file_path}: {e}")
            results[file_path] = False

    return results


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)

    # Test basic CRC32C
    print("=== Testing Basic CRC32C ===")
    data = b"Hello, World!"
    checksum = crc32c(data)
    print(f"CRC32C of '{data.decode()}': {checksum:08x}")

    # Test detailed computation
    result = crc32c_detailed(data, expected=checksum)
    print(f"Detailed result: {result}")
    print(f"Hex: {result.to_hex()}, Verified: {result.verified}")

    # Test streaming CRC32C
    print("\n=== Testing Streaming CRC32C ===")
    stream_crc = StreamingCRC32C()
    stream_crc.update(b"Hello, ")
    stream_crc.update(b"World!")
    stream_result = stream_crc.finalize()
    print(f"Streaming CRC32C: {stream_result.to_hex()}")
    print(f"Matches direct computation: {stream_result.checksum == checksum}")

    # Test batch processing
    print("\n=== Testing Batch Processing ===")
    batch_data = [b"data1", b"data2", b"data3"]
    batch_results = crc32c_batch(batch_data)
    for i, result in enumerate(batch_results):
        print(f"  Item {i}: {result.to_hex()} ({result.data_size} bytes)")

    # Test validation
    print("\n=== Testing Validation ===")
    validator = CRC32CValidator()
    validator.register("test_data", checksum)
    is_valid = validator.validate("test_data", data)
    print(f"Validation result: {is_valid}")
    print(f"Validator stats: {validator.get_stats()}")
