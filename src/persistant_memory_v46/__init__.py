"""
Vulcan Persistent Memory v46 - Advanced Memory System

A sophisticated persistent memory system with:
- Graph-based RAG with multi-level indexing and hybrid retrieval
- Merkle LSM tree for efficient storage and compaction
- S3/CloudFront packfile storage with compression and encryption
- Zero-knowledge proofs for privacy-preserving unlearning
- Advanced machine unlearning with multiple algorithms

Author: Vulcan LLM
Version: 46
"""

import logging
from .graph_rag import GraphNode, GraphRAG, RetrievalResult
from .lsm import BloomFilter, MerkleLSM, MerkleLSMDAG, Packfile
from .store import PackfileStore
from .unlearning import GradientSurgeryUnlearner, UnlearningEngine
from .zk import GrothProof, MerkleTree, ZKCircuit, ZKProver

__version__ = "46.0.0"
__author__ = "Vulcan LLM"

__all__ = [
    # Storage
    "PackfileStore",
    # LSM Tree
    "MerkleLSM",
    "BloomFilter",
    "Packfile",
    "MerkleLSMDAG",
    # Graph RAG
    "GraphRAG",
    "RetrievalResult",
    "GraphNode",
    # Unlearning
    "UnlearningEngine",
    "GradientSurgeryUnlearner",
    # Zero-Knowledge
    "ZKProver",
    "ZKCircuit",
    "GrothProof",
    "MerkleTree",
]


def create_memory_system(
    s3_bucket: str,
    embedding_model: str = "llm_embeddings",
    compression: str = "zstd",
    encryption: str = "AES256",
    **kwargs,
) -> dict:
    """
    Create a complete memory system with all components.

    Args:
        s3_bucket: S3 bucket for storage
        embedding_model: Embedding model to use
        compression: Compression algorithm (zstd, zlib, lz4)
        encryption: Encryption algorithm (AES256, aws:kms)
        **kwargs: Additional configuration

    Returns:
        Dictionary containing all initialized components
    """
    # Initialize storage
    store = PackfileStore(
        s3_bucket=s3_bucket,
        compression=compression,
        encryption=encryption,
        **{k: v for k, v in kwargs.items() if k.startswith("storage_")},
    )

    # Initialize LSM tree
    lsm = MerkleLSM(
        packfile_size_mb=kwargs.get("packfile_size_mb", 32),
        compaction_strategy=kwargs.get("compaction_strategy", "adaptive"),
        bloom_filter=kwargs.get("bloom_filter", True),
        **{k: v for k, v in kwargs.items() if k.startswith("lsm_")},
    )

    # Initialize Graph RAG
    graph_rag = GraphRAG(
        embedding_model=embedding_model,
        index_type=kwargs.get("index_type", "disk_based_tier_c"),
        prefetch=kwargs.get("prefetch", True),
        **{k: v for k, v in kwargs.items() if k.startswith("rag_")},
    )

    # Initialize Unlearning Engine
    unlearning = UnlearningEngine(
        merkle_graph=lsm.dag,
        method=kwargs.get("unlearning_method", "gradient_surgery"),
        enable_verification=kwargs.get("enable_verification", True),
        **{k: v for k, v in kwargs.items() if k.startswith("unlearning_")},
    )

    # Initialize ZK Prover
    zk_prover = ZKProver(
        circuit_hash=kwargs.get("circuit_hash", "sha256:unlearning_v1.0"),
        proof_system=kwargs.get("proof_system", "groth16"),
        **{k: v for k, v in kwargs.items() if k.startswith("zk_")},
    )

    return {
        "store": store,
        "lsm": lsm,
        "graph_rag": graph_rag,
        "unlearning": unlearning,
        "zk_prover": zk_prover,
        "version": __version__,
    }


def get_system_info() -> dict:
    """
    Get information about the memory system.

    Returns:
        Dictionary with system information
    """
    return {
        "version": __version__,
        "author": __author__,
        "components": {
            "storage": "S3/CloudFront with compression and encryption",
            "lsm": "Merkle LSM tree with adaptive compaction",
            "graph_rag": "Hybrid retrieval with graph expansion",
            "unlearning": "Multiple algorithms (gradient surgery, SISA, influence)",
            "zk": "Zero-knowledge proofs (Groth16, PLONK)",
        },
        "features": [
            "Multi-level indexing",
            "Bloom filters",
            "Adaptive range requests",
            "Background compaction",
            "Privacy-preserving unlearning",
            "Cryptographic verification",
            "Async operations",
            "Comprehensive statistics",
        ],
    }


# Module-level convenience functions


def quick_start(s3_bucket: str, **kwargs):
    """
    Quick start with sensible defaults.

    Args:
        s3_bucket: S3 bucket for storage
        **kwargs: Optional overrides

    Returns:
        Initialized memory system
    """
    defaults = {
        "compression": "zstd",
        "encryption": "AES256",
        "embedding_model": "llm_embeddings",
        "packfile_size_mb": 32,
        "compaction_strategy": "adaptive",
        "bloom_filter": True,
        "index_type": "disk_based_tier_c",
        "prefetch": True,
        "unlearning_method": "gradient_surgery",
        "enable_verification": True,
        "proof_system": "groth16",
    }
    defaults.update(kwargs)

    return create_memory_system(s3_bucket, **defaults)


# Package initialization

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)
logger.info(f"Vulcan Persistent Memory v{__version__} loaded")
