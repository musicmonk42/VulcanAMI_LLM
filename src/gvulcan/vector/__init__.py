"""
Vulcan Vector Module - Milvus Integration

This module provides integration with Milvus vector database for the
Vulcan Memory System, including:
- Collection bootstrapping and management
- Vector quantization (FP16, INT8, binary)
- Client operations for search and insert

Components:
- milvus_bootstrap: Collection initialization and schema management
- milvus_client: Client wrapper for Milvus operations
- quantization: Vector quantization utilities
"""

from .milvus_bootstrap import (
    bootstrap_all_collections,
    create_collection_if_not_exists,
    get_collection_stats,
    load_collection,
    validate_config,
)
from .milvus_client import MilvusIndex
from .quantization import (
    dequantize_fp16,
    dequantize_int8,
    quantize_binary,
    quantize_fp16,
    quantize_int8,
)

__all__ = [
    # Bootstrap
    "bootstrap_all_collections",
    "create_collection_if_not_exists",
    "get_collection_stats",
    "load_collection",
    "validate_config",
    # Client
    "MilvusIndex",
    # Quantization
    "quantize_fp16",
    "dequantize_fp16",
    "quantize_int8",
    "dequantize_int8",
    "quantize_binary",
]
