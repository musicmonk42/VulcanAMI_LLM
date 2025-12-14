"""
Vector Quantization Methods

This module provides comprehensive vector quantization techniques for efficient
storage and similarity search, including rotational quantization, product quantization,
and error-correcting codes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QuantizationMetadata:
    """
    Metadata for quantized vectors.

    Attributes:
        method: Quantization method name
        original_dim: Original vector dimensionality
        quantized_dim: Quantized dimensionality
        bits_per_component: Bits per component
        compression_ratio: Achieved compression ratio
        parameters: Method-specific parameters
    """

    method: str
    original_dim: int
    quantized_dim: int
    bits_per_component: int
    compression_ratio: float
    parameters: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "method": self.method,
            "original_dim": self.original_dim,
            "quantized_dim": self.quantized_dim,
            "bits_per_component": self.bits_per_component,
            "compression_ratio": self.compression_ratio,
            "parameters": self.parameters,
        }


def rotational_8bit(
    fp16: np.ndarray, use_optimal_rotation: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Rotational 8-bit quantization for vectors.

    This method applies an optional rotation to align principal components,
    then quantizes to 8-bit integers with per-row scaling.

    Args:
        fp16: Input float16 vectors (N x D)
        use_optimal_rotation: Whether to compute optimal rotation matrix

    Returns:
        Tuple of (quantized codes as int8, metadata dict)

    Example:
        >>> vectors = np.random.randn(100, 768).astype(np.float16)
        >>> codes, meta = rotational_8bit(vectors)
        >>> print(f"Compression: {meta['compression_ratio']:.2f}x")
    """
    if fp16.ndim == 1:
        fp16 = fp16.reshape(1, -1)

    n, d = fp16.shape
    rotated = fp16.astype(np.float32)
    rotation_matrix = None

    # Optional: compute and apply rotation for better quantization
    if use_optimal_rotation and n > 1:
        try:
            # PCA-based rotation to align with principal components
            mean = np.mean(rotated, axis=0, keepdims=True)
            centered = rotated - mean

            # Compute covariance
            cov = np.cov(centered.T)

            # Eigen decomposition
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            # Sort by eigenvalue (descending)
            idx = eigenvalues.argsort()[::-1]
            rotation_matrix = eigenvectors[:, idx]

            # Apply rotation
            rotated = centered @ rotation_matrix

            logger.debug(f"Applied rotational quantization with PCA")
        except Exception as e:
            logger.warning(f"Failed to compute rotation matrix: {e}, using identity")

    # Compute per-row scale factors
    scale = np.maximum(np.abs(rotated).max(axis=1, keepdims=True), 1e-8)

    # Quantize to 8-bit
    codes = np.clip((rotated / scale) * 127.0, -128, 127).astype(np.int8)

    # Compute compression ratio
    original_bytes = fp16.nbytes
    quantized_bytes = codes.nbytes + scale.nbytes
    if rotation_matrix is not None:
        quantized_bytes += rotation_matrix.nbytes
    compression_ratio = original_bytes / quantized_bytes

    # Metadata
    meta = {
        "scale": scale.squeeze().tolist() if scale.size < 1000 else scale.mean(),
        "has_rotation": rotation_matrix is not None,
        "rotation_matrix": (
            rotation_matrix.tolist()
            if rotation_matrix is not None and rotation_matrix.size < 10000
            else None
        ),
        "compression_ratio": compression_ratio,
        "quantization_range": [-128, 127],
        "method": "rotational_8bit",
    }

    logger.debug(f"Rotational 8-bit quantization: {compression_ratio:.2f}x compression")

    return codes, meta


def dequantize_rotational_8bit(codes: np.ndarray, meta: Dict[str, Any]) -> np.ndarray:
    """
    Dequantize rotational 8-bit codes back to float.

    Args:
        codes: Quantized int8 codes
        meta: Metadata from quantization

    Returns:
        Dequantized float32 vectors
    """
    # Convert to float and descale
    scale = np.array(meta["scale"])
    if scale.ndim == 0:
        scale = scale.reshape(1, 1)
    elif scale.ndim == 1:
        scale = scale.reshape(-1, 1)

    dequantized = (codes.astype(np.float32) / 127.0) * scale

    # Reverse rotation if applied
    if meta.get("has_rotation") and meta.get("rotation_matrix") is not None:
        rotation_matrix = np.array(meta["rotation_matrix"])
        dequantized = dequantized @ rotation_matrix.T

    return dequantized


def int4_with_ecc(
    fp16: np.ndarray, ecc_bits: int = 2
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    4-bit quantization with error-correcting codes.

    Quantizes vectors to 4-bit integers and adds ECC parity bits for
    error detection and correction.

    Args:
        fp16: Input float16 vectors (N x D)
        ecc_bits: Number of ECC parity bits per block

    Returns:
        Tuple of (packed uint8 array, metadata dict)

    Example:
        >>> vectors = np.random.randn(100, 768).astype(np.float16)
        >>> codes, meta = int4_with_ecc(vectors)
        >>> print(f"ECC: {meta['ecc_overhead']:.2%} overhead")
    """
    if fp16.ndim == 1:
        fp16 = fp16.reshape(1, -1)

    n, d = fp16.shape

    # Normalize to [-1, 1] range
    max_val = np.abs(fp16).max(axis=1, keepdims=True)
    max_val = np.maximum(max_val, 1e-8)
    normalized = fp16 / max_val

    # Quantize to 4-bit [-7, 7]
    scaled = np.clip(normalized * 7.0, -8, 7).astype(np.int8)

    # Pack two 4-bit values into each uint8
    packed = np.zeros((n, (d + 1) // 2), dtype=np.uint8)

    for i in range(d):
        v = (scaled[:, i] & 0x0F).astype(np.uint8)
        if i % 2 == 0:
            packed[:, i // 2] |= v
        else:
            packed[:, i // 2] |= v << 4

    # Compute ECC parity bits (simple parity for demonstration)
    # In production, use Hamming codes or Reed-Solomon
    ecc_parity = []
    for row in packed:
        # Simple parity: XOR of all bytes
        parity = 0
        for byte in row:
            parity ^= byte
        ecc_parity.append(parity)

    ecc_parity = np.array(ecc_parity, dtype=np.uint8)

    # Compute compression
    original_bytes = fp16.nbytes
    quantized_bytes = packed.nbytes + ecc_parity.nbytes + max_val.nbytes
    compression_ratio = original_bytes / quantized_bytes
    ecc_overhead = ecc_parity.nbytes / packed.nbytes

    meta = {
        "ecc": "xor_parity",
        "ecc_parity": ecc_parity.tolist(),
        "ecc_bits": ecc_bits,
        "ecc_overhead": ecc_overhead,
        "scale": max_val.squeeze().tolist() if max_val.size < 1000 else max_val.mean(),
        "compression_ratio": compression_ratio,
        "quantization_range": [-8, 7],
        "method": "int4_with_ecc",
    }

    logger.debug(
        f"INT4 with ECC quantization: {compression_ratio:.2f}x compression, "
        f"ECC overhead: {ecc_overhead:.2%}"
    )

    return packed, meta


def dequantize_int4_with_ecc(
    packed: np.ndarray, meta: Dict[str, Any], original_dim: int, verify_ecc: bool = True
) -> np.ndarray:
    """
    Dequantize 4-bit ECC codes back to float.

    Args:
        packed: Packed uint8 array
        meta: Metadata from quantization
        original_dim: Original vector dimensionality
        verify_ecc: Whether to verify ECC parity

    Returns:
        Dequantized float32 vectors
    """
    n = packed.shape[0]

    # Verify ECC if requested
    if verify_ecc and "ecc_parity" in meta:
        stored_parity = np.array(meta["ecc_parity"], dtype=np.uint8)

        for i, row in enumerate(packed):
            computed_parity = 0
            for byte in row:
                computed_parity ^= byte

            if i < len(stored_parity) and computed_parity != stored_parity[i]:
                logger.warning(
                    f"ECC parity mismatch for row {i}, data may be corrupted"
                )

    # Unpack 4-bit values
    scaled = np.zeros((n, original_dim), dtype=np.int8)

    for i in range(original_dim):
        if i % 2 == 0:
            # Lower 4 bits
            v = (packed[:, i // 2] & 0x0F).astype(np.int8)
        else:
            # Upper 4 bits
            v = ((packed[:, i // 2] >> 4) & 0x0F).astype(np.int8)

        # Sign extend from 4-bit to 8-bit
        v = np.where(v >= 8, v - 16, v)
        scaled[:, i] = v

    # Dequantize
    scale = np.array(meta["scale"])
    if scale.ndim == 0:
        scale = scale.reshape(1, 1)
    elif scale.ndim == 1:
        scale = scale.reshape(-1, 1)

    dequantized = (scaled.astype(np.float32) / 7.0) * scale

    return dequantized


def product_quantization(
    vectors: np.ndarray, n_subspaces: int = 8, n_centroids: int = 256
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Product Quantization (PQ) for vectors.

    Divides vectors into subspaces and quantizes each subspace independently
    using k-means clustering.

    Args:
        vectors: Input vectors (N x D)
        n_subspaces: Number of subspaces (D must be divisible)
        n_centroids: Number of centroids per subspace (typically 256 for 8-bit)

    Returns:
        Tuple of (quantized codes, metadata with codebooks)
    """
    n, d = vectors.shape

    if d % n_subspaces != 0:
        raise ValueError(
            f"Dimension {d} must be divisible by n_subspaces {n_subspaces}"
        )

    subspace_dim = d // n_subspaces
    codes = np.zeros((n, n_subspaces), dtype=np.uint8)
    codebooks = []

    # Quantize each subspace
    for i in range(n_subspaces):
        start_idx = i * subspace_dim
        end_idx = (i + 1) * subspace_dim
        subspace = vectors[:, start_idx:end_idx]

        # Simple k-means (in production, use sklearn or faiss)
        # For now, just sample centroids
        indices = np.random.choice(n, min(n_centroids, n), replace=False)
        centroids = subspace[indices]

        # Assign to nearest centroid
        distances = np.linalg.norm(
            subspace[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2
        )
        codes[:, i] = np.argmin(distances, axis=1).astype(np.uint8)

        codebooks.append(centroids)

    # Compute compression
    original_bytes = vectors.nbytes
    quantized_bytes = codes.nbytes + sum(cb.nbytes for cb in codebooks)
    compression_ratio = original_bytes / quantized_bytes

    meta = {
        "n_subspaces": n_subspaces,
        "subspace_dim": subspace_dim,
        "n_centroids": n_centroids,
        "codebooks": [cb.tolist() for cb in codebooks],
        "compression_ratio": compression_ratio,
        "method": "product_quantization",
    }

    logger.debug(
        f"Product quantization: {n_subspaces} subspaces, "
        f"{n_centroids} centroids, {compression_ratio:.2f}x compression"
    )

    return codes, meta


def dequantize_product_quantization(
    codes: np.ndarray, meta: Dict[str, Any]
) -> np.ndarray:
    """
    Dequantize product quantization codes.

    Args:
        codes: Quantized codes (N x n_subspaces)
        meta: Metadata with codebooks

    Returns:
        Reconstructed vectors
    """
    n, n_subspaces = codes.shape
    codebooks = [np.array(cb) for cb in meta["codebooks"]]
    subspace_dim = meta["subspace_dim"]

    reconstructed = np.zeros((n, n_subspaces * subspace_dim), dtype=np.float32)

    for i in range(n_subspaces):
        start_idx = i * subspace_dim
        end_idx = (i + 1) * subspace_dim

        # Lookup centroids
        centroid_indices = codes[:, i]
        reconstructed[:, start_idx:end_idx] = codebooks[i][centroid_indices]

    return reconstructed


def binary_quantization(
    vectors: np.ndarray, method: str = "sign"
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Binary quantization for extreme compression.

    Args:
        vectors: Input vectors (N x D)
        method: Quantization method ("sign" or "threshold")

    Returns:
        Tuple of (packed binary codes, metadata)
    """
    n, d = vectors.shape

    if method == "sign":
        # Simple sign-based binarization
        binary = (vectors > 0).astype(np.uint8)
    elif method == "threshold":
        # Threshold at median
        threshold = np.median(vectors, axis=1, keepdims=True)
        binary = (vectors > threshold).astype(np.uint8)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Pack bits
    packed = np.packbits(binary, axis=1)

    compression_ratio = vectors.nbytes / packed.nbytes

    meta = {
        "method": f"binary_{method}",
        "compression_ratio": compression_ratio,
        "original_dim": d,
    }

    logger.debug(f"Binary quantization: {compression_ratio:.2f}x compression")

    return packed, meta


def adaptive_quantization(
    vectors: np.ndarray, target_bits: int = 4, per_channel: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Adaptive quantization with learned scaling factors.

    Args:
        vectors: Input vectors (N x D)
        target_bits: Target bits per value
        per_channel: Whether to use per-channel scaling

    Returns:
        Tuple of (quantized codes, metadata)
    """
    n, d = vectors.shape

    if per_channel:
        # Per-channel min-max scaling
        vmin = vectors.min(axis=0, keepdims=True)
        vmax = vectors.max(axis=0, keepdims=True)
    else:
        # Global min-max
        vmin = vectors.min()
        vmax = vectors.max()

    # Scale to [0, 2^target_bits - 1]
    max_val = (1 << target_bits) - 1
    scaled = (vectors - vmin) / (vmax - vmin + 1e-8) * max_val

    # Quantize
    if target_bits <= 8:
        codes = np.clip(scaled, 0, max_val).astype(np.uint8)
    else:
        codes = np.clip(scaled, 0, max_val).astype(np.uint16)

    compression_ratio = vectors.nbytes / (codes.nbytes + vmin.nbytes + vmax.nbytes)

    meta = {
        "vmin": vmin.tolist() if vmin.size < 1000 else float(vmin.mean()),
        "vmax": vmax.tolist() if vmax.size < 1000 else float(vmax.mean()),
        "target_bits": target_bits,
        "per_channel": per_channel,
        "compression_ratio": compression_ratio,
        "method": "adaptive",
    }

    return codes, meta


def scalar_quantization(
    vectors: np.ndarray, bits: int = 8
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Scalar quantization for vectors.

    Simple and efficient quantization that maps each component independently
    to a fixed-point representation using uniform quantization bins.
    This is the most basic form of quantization, similar to rotational_8bit
    but without the rotation step.

    Args:
        vectors: Input vectors (N x D)
        bits: Number of bits per component (4, 8, or 16)

    Returns:
        Tuple of (quantized codes, metadata dict)

    Example:
        >>> vectors = np.random.randn(100, 768).astype(np.float32)
        >>> codes, meta = scalar_quantization(vectors, bits=8)
        >>> print(f"Compression: {meta['compression_ratio']:.2f}x")

    Note:
        - bits=4: Uses int8 with 4-bit packing
        - bits=8: Uses int8 directly
        - bits=16: Uses int16
    """
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)

    n, d = vectors.shape

    # Validate bits parameter
    if bits not in [4, 8, 16]:
        raise ValueError(f"bits must be 4, 8, or 16, got {bits}")

    # Compute per-vector scale factors (min-max scaling)
    vmin = vectors.min(axis=1, keepdims=True)
    vmax = vectors.max(axis=1, keepdims=True)

    # Avoid division by zero
    scale_range = vmax - vmin
    scale_range = np.maximum(scale_range, 1e-8)

    # Determine quantization range based on bits
    if bits == 4:
        # 4-bit: [-7, 7] range
        qmin, qmax = -7, 7
        dtype = np.int8
    elif bits == 8:
        # 8-bit: [-128, 127] range
        qmin, qmax = -128, 127
        dtype = np.int8
    else:  # bits == 16
        # 16-bit: [-32768, 32767] range
        qmin, qmax = -32768, 32767
        dtype = np.int16

    # Normalize to [0, 1] then scale to quantization range
    normalized = (vectors - vmin) / scale_range
    scaled = normalized * (qmax - qmin) + qmin

    # Quantize
    codes = np.clip(scaled, qmin, qmax).astype(dtype)

    # For 4-bit, pack two values per byte
    if bits == 4:
        packed = np.zeros((n, (d + 1) // 2), dtype=np.uint8)
        for i in range(d):
            v = (codes[:, i] & 0x0F).astype(np.uint8)
            if i % 2 == 0:
                packed[:, i // 2] |= v
            else:
                packed[:, i // 2] |= v << 4
        codes = packed

    # Compute compression ratio
    original_bytes = vectors.nbytes
    quantized_bytes = codes.nbytes + vmin.nbytes + vmax.nbytes
    compression_ratio = original_bytes / quantized_bytes

    # Metadata
    meta = {
        "method": "scalar_quantization",
        "bits": bits,
        "vmin": vmin.squeeze().tolist() if vmin.size < 1000 else float(vmin.mean()),
        "vmax": vmax.squeeze().tolist() if vmax.size < 1000 else float(vmax.mean()),
        "quantization_range": [int(qmin), int(qmax)],
        "compression_ratio": compression_ratio,
        "original_dim": d,
    }

    logger.debug(
        f"Scalar quantization ({bits}-bit): {compression_ratio:.2f}x compression"
    )

    return codes, meta


def dequantize_scalar_quantization(
    codes: np.ndarray, meta: Dict[str, Any]
) -> np.ndarray:
    """
    Dequantize scalar quantization codes back to float.

    Args:
        codes: Quantized codes
        meta: Metadata from quantization

    Returns:
        Dequantized float32 vectors
    """
    bits = meta["bits"]
    original_dim = meta["original_dim"]
    vmin = np.array(meta["vmin"])
    vmax = np.array(meta["vmax"])
    qmin, qmax = meta["quantization_range"]

    # Ensure proper shape for broadcasting
    if vmin.ndim == 0:
        vmin = vmin.reshape(1, 1)
    elif vmin.ndim == 1:
        vmin = vmin.reshape(-1, 1)

    if vmax.ndim == 0:
        vmax = vmax.reshape(1, 1)
    elif vmax.ndim == 1:
        vmax = vmax.reshape(-1, 1)

    # Unpack 4-bit if needed
    if bits == 4:
        n = codes.shape[0]
        unpacked = np.zeros((n, original_dim), dtype=np.int8)

        for i in range(original_dim):
            if i % 2 == 0:
                v = (codes[:, i // 2] & 0x0F).astype(np.int8)
            else:
                v = ((codes[:, i // 2] >> 4) & 0x0F).astype(np.int8)

            # Sign extend from 4-bit to 8-bit
            v = np.where(v >= 8, v - 16, v)
            unpacked[:, i] = v

        codes = unpacked

    # Dequantize: map from quantization range to [0, 1] to original range
    scale_range = vmax - vmin
    normalized = (codes.astype(np.float32) - qmin) / (qmax - qmin)
    dequantized = normalized * scale_range + vmin

    return dequantized
