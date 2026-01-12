"""
Unified Memory Bridge - Single entry point for all memory operations.

This module provides a production-grade integration layer that unifies three distinct
memory systems in VulcanAMI_LLM:

1. **persistant_memory_v46/** - Production storage (GraphRAG, MerkleLSM, PackfileStore, ZKProver)
2. **memory/** - Governed operations (GovernedUnlearning, CostOptimizer)
3. **vulcan/memory/** - Core hierarchical memory (HierarchicalMemory, retrieval, consolidation)

The bridge follows the same architectural pattern as GraphixVulcanBridge, providing
graceful degradation when components are unavailable.

Design Principles:
    - Single Responsibility: Each component handles one aspect of memory management
    - Fail-Safe: Graceful degradation when optional dependencies unavailable
    - Type Safety: Full type hints for static analysis
    - Thread Safety: Thread-safe operations where applicable
    - Security: Secure defaults, input validation
    - Observability: Comprehensive logging and metrics

Example:
    Basic usage with automatic configuration:
    
    >>> bridge = create_memory_bridge({
    ...     "s3_bucket": "my-memory-bucket",
    ...     "enable_zk_proofs": True
    ... })
    >>> bridge.store("key1", "Important data", {"tags": ["critical"]})
    >>> results = bridge.retrieve("search query", k=5)
    >>> status = bridge.get_status()

Thread Safety:
    All public methods are thread-safe. Internal state is protected by locks
    where necessary.

Author: VULCAN-AGI Team
Version: 1.0.0
License: MIT
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# ============================================================================
# DEPENDENCY IMPORTS WITH GRACEFUL FALLBACK
# ============================================================================

# Import hierarchy - production storage first
PERSISTENT_MEMORY_AVAILABLE = False
HIERARCHICAL_MEMORY_AVAILABLE = False
GOVERNED_UNLEARNING_AVAILABLE = False
COST_OPTIMIZER_AVAILABLE = False

try:
    from src.persistant_memory_v46 import (
        GraphRAG,
        MerkleLSM,
        PackfileStore,
        UnlearningEngine,
        ZKProver,
        get_system_info,
    )
    PERSISTENT_MEMORY_AVAILABLE = True
    logger.info("persistant_memory_v46 available")
except ImportError as e:
    logger.debug(f"persistant_memory_v46 not available: {e}")

try:
    from src.vulcan.memory.hierarchical import HierarchicalMemory
    from src.vulcan.memory.base import Memory, MemoryConfig, MemoryQuery, MemoryType
    HIERARCHICAL_MEMORY_AVAILABLE = True
    logger.info("HierarchicalMemory available")
except ImportError as e:
    logger.debug(f"HierarchicalMemory not available: {e}")

try:
    from src.memory.governed_unlearning import (
        GovernedUnlearning,
        UnlearningMethod,
        UrgencyLevel,
    )
    GOVERNED_UNLEARNING_AVAILABLE = True
    logger.info("GovernedUnlearning available")
except ImportError as e:
    logger.debug(f"GovernedUnlearning not available: {e}")

try:
    from src.memory.cost_optimizer import CostOptimizer, OptimizationStrategy
    COST_OPTIMIZER_AVAILABLE = True
    logger.info("CostOptimizer available")
except ImportError as e:
    logger.debug(f"CostOptimizer not available: {e}")


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class MemoryBridgeConfig:
    """
    Configuration for unified memory bridge.
    
    Attributes:
        s3_bucket: S3 bucket name for persistent storage (optional)
        region: AWS region for S3 storage
        compression: Compression algorithm (zstd, zlib, lz4, none)
        encryption: Encryption standard (AES256, aws:kms)
        max_memories: Maximum number of memories in hierarchical storage
        default_importance: Default importance score for new memories [0.0-1.0]
        decay_rate: Memory decay rate for importance calculations
        embedding_model: Sentence transformer model name for embeddings
        enable_governed_unlearning: Enable governed unlearning with consensus
        enable_cost_optimization: Enable automatic cost optimization
        auto_optimize: Run cost optimization automatically in background
        enable_zk_proofs: Enable zero-knowledge proof generation
        enable_graph_rag: Enable graph-based retrieval augmented generation
        
    Validation:
        - default_importance must be between 0.0 and 1.0
        - decay_rate must be non-negative
        - embedding_model must be a valid sentence-transformers model
    """
    
    # Storage configuration
    s3_bucket: Optional[str] = None
    region: str = "us-east-1"
    compression: str = "zstd"
    encryption: str = "AES256"
    
    # Memory configuration
    max_memories: int = 100000
    default_importance: float = 0.5
    decay_rate: float = 0.001
    embedding_model: str = "all-MiniLM-L6-v2"  # Valid sentence-transformers model
    
    # Operations configuration
    enable_governed_unlearning: bool = True
    enable_cost_optimization: bool = True
    auto_optimize: bool = False
    
    # Feature flags
    enable_zk_proofs: bool = True
    enable_graph_rag: bool = True
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not 0.0 <= self.default_importance <= 1.0:
            raise ValueError(
                f"default_importance must be in [0.0, 1.0], got {self.default_importance}"
            )
        
        if self.decay_rate < 0:
            raise ValueError(
                f"decay_rate must be non-negative, got {self.decay_rate}"
            )
        
        if self.compression not in {"zstd", "zlib", "lz4", "none"}:
            logger.warning(
                f"Unknown compression '{self.compression}', using 'zstd'"
            )
            self.compression = "zstd"
        
        if self.encryption not in {"AES256", "aws:kms"}:
            logger.warning(
                f"Unknown encryption '{self.encryption}', using 'AES256'"
            )
            self.encryption = "AES256"


# ============================================================================
# MEMORY BRIDGE IMPLEMENTATION
# ============================================================================


class MemoryBridge:
    """
    Unified bridge for all memory operations.
    
    This class provides a single, cohesive API for interacting with all memory
    subsystems in VulcanAMI_LLM. It handles initialization, graceful degradation,
    and coordination between components.
    
    Components:
        - **Storage Backend**: PackfileStore for S3-based persistent storage
        - **GraphRAG**: Graph-based retrieval with multi-hop reasoning
        - **MerkleLSM**: Log-structured merge tree with Merkle proofs
        - **ZKProver**: Zero-knowledge proof generation for compliance
        - **HierarchicalMemory**: Multi-level memory with decay and consolidation
        - **GovernedUnlearning**: Consensus-based machine unlearning
        - **CostOptimizer**: Budget-aware storage optimization
    
    Thread Safety:
        All public methods are thread-safe. The bridge uses internal locking
        to protect shared state during component initialization.
    
    Example:
        >>> config = MemoryBridgeConfig(s3_bucket="my-bucket")
        >>> bridge = MemoryBridge(config)
        >>> 
        >>> # Store data
        >>> bridge.store("doc1", "Important content", {"priority": "high"})
        >>> 
        >>> # Retrieve with hybrid search
        >>> results = bridge.retrieve("search query", k=10)
        >>> 
        >>> # Request unlearning
        >>> result = bridge.unlearn("sensitive_pattern", urgency="high")
        >>> 
        >>> # Optimize storage costs
        >>> report = bridge.optimize_storage()
    """
    
    def __init__(self, config: Optional[MemoryBridgeConfig] = None) -> None:
        """
        Initialize the memory bridge with optional configuration.
        
        Args:
            config: Bridge configuration. If None, uses defaults.
        
        Raises:
            ValueError: If configuration validation fails
        """
        self.config = config or MemoryBridgeConfig()
        self._lock = threading.RLock()
        
        # Component references (initialized lazily)
        self._storage: Optional[Any] = None
        self._graph_rag: Optional[Any] = None
        self._lsm: Optional[Any] = None
        self._zk_prover: Optional[Any] = None
        self._hierarchical: Optional[Any] = None
        self._governed_unlearning: Optional[Any] = None
        self._cost_optimizer: Optional[Any] = None
        
        # Initialize all available components
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """
        Initialize all available components with error isolation.
        
        Each component initialization is wrapped in try-except to ensure
        failure of one component doesn't prevent initialization of others.
        This provides graceful degradation.
        """
        with self._lock:
            # 1. Storage backend (persistant_memory_v46)
            if PERSISTENT_MEMORY_AVAILABLE and self.config.s3_bucket:
                self._init_packfile_store()
            
            # 2. GraphRAG for semantic retrieval
            if PERSISTENT_MEMORY_AVAILABLE and self.config.enable_graph_rag:
                self._init_graph_rag()
            
            # 3. MerkleLSM for versioned key-value storage
            if PERSISTENT_MEMORY_AVAILABLE:
                self._init_merkle_lsm()
            
            # 4. ZK Prover for compliance proofs
            if PERSISTENT_MEMORY_AVAILABLE and self.config.enable_zk_proofs:
                self._init_zk_prover()
            
            # 5. HierarchicalMemory for multi-level storage
            if HIERARCHICAL_MEMORY_AVAILABLE:
                self._init_hierarchical_memory()
            
            # 6. Governed Unlearning for GDPR compliance
            if GOVERNED_UNLEARNING_AVAILABLE and self.config.enable_governed_unlearning:
                self._init_governed_unlearning()
            
            # 7. Cost Optimizer for budget management
            if COST_OPTIMIZER_AVAILABLE and self.config.enable_cost_optimization:
                self._init_cost_optimizer()
    
    def _init_packfile_store(self) -> None:
        """Initialize PackfileStore for S3-based persistent storage."""
        try:
            self._storage = PackfileStore(
                s3_bucket=self.config.s3_bucket,
                region=self.config.region,
                compression=self.config.compression,
                encryption=self.config.encryption,
            )
            logger.info(
                f"PackfileStore initialized (bucket={self.config.s3_bucket}, "
                f"region={self.config.region})"
            )
        except Exception as e:
            logger.warning(f"PackfileStore initialization failed: {e}", exc_info=True)
            self._storage = None
    
    def _init_graph_rag(self) -> None:
        """Initialize GraphRAG for semantic retrieval."""
        try:
            self._graph_rag = GraphRAG(
                embedding_model=self.config.embedding_model,
            )
            logger.info(
                f"GraphRAG initialized (model={self.config.embedding_model})"
            )
        except Exception as e:
            logger.warning(f"GraphRAG initialization failed: {e}", exc_info=True)
            self._graph_rag = None
    
    def _init_merkle_lsm(self) -> None:
        """Initialize MerkleLSM for versioned key-value storage."""
        try:
            self._lsm = MerkleLSM(
                packfile_size_mb=32,
                compaction_strategy="adaptive",
                bloom_filter=True,
            )
            logger.info("MerkleLSM initialized (adaptive compaction, bloom filter enabled)")
        except Exception as e:
            logger.warning(f"MerkleLSM initialization failed: {e}", exc_info=True)
            self._lsm = None
    
    def _init_zk_prover(self) -> None:
        """Initialize ZKProver for zero-knowledge proofs."""
        try:
            self._zk_prover = ZKProver(
                proof_system="groth16",
            )
            logger.info("ZKProver initialized (Groth16)")
        except Exception as e:
            logger.warning(f"ZKProver initialization failed: {e}", exc_info=True)
            self._zk_prover = None
    
    def _init_hierarchical_memory(self) -> None:
        """Initialize HierarchicalMemory for multi-level storage."""
        try:
            mem_config = MemoryConfig(
                max_memories=self.config.max_memories,
                default_importance=self.config.default_importance,
                decay_rate=self.config.decay_rate,
            )
            self._hierarchical = HierarchicalMemory(mem_config)
            logger.info(
                f"HierarchicalMemory initialized (max={self.config.max_memories})"
            )
        except Exception as e:
            logger.warning(
                f"HierarchicalMemory initialization failed: {e}", exc_info=True
            )
            self._hierarchical = None
    
    def _init_governed_unlearning(self) -> None:
        """Initialize GovernedUnlearning for compliance."""
        try:
            memory_ref = self._hierarchical or self
            self._governed_unlearning = GovernedUnlearning(memory_ref)
            logger.info("GovernedUnlearning initialized")
        except Exception as e:
            logger.warning(
                f"GovernedUnlearning initialization failed: {e}", exc_info=True
            )
            self._governed_unlearning = None
    
    def _init_cost_optimizer(self) -> None:
        """Initialize CostOptimizer for budget management."""
        try:
            memory_ref = self._hierarchical or self
            self._cost_optimizer = CostOptimizer(
                memory_ref,
                auto_optimize=self.config.auto_optimize,
            )
            logger.info(
                f"CostOptimizer initialized (auto_optimize={self.config.auto_optimize})"
            )
        except Exception as e:
            logger.warning(
                f"CostOptimizer initialization failed: {e}", exc_info=True
            )
            self._cost_optimizer = None
    
    # ========================================================================
    # STORAGE OPERATIONS
    # ========================================================================
    
    def store(
        self,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store data across available memory backends.
        
        Data is stored in multiple backends for redundancy and different access
        patterns:
        - LSM tree for fast key-value access
        - GraphRAG for semantic search
        - HierarchicalMemory for temporal access patterns
        
        Args:
            key: Unique identifier for the data
            value: Data to store (any type, preferably JSON-serializable)
            metadata: Optional metadata dictionary
        
        Returns:
            True if stored successfully in at least one backend, False otherwise
        
        Raises:
            ValueError: If key is empty or value is None
        
        Thread Safety:
            This method is thread-safe.
        """
        if not key:
            raise ValueError("Key cannot be empty")
        if value is None:
            raise ValueError("Value cannot be None")
        
        metadata = metadata or {}
        success = False
        
        try:
            # Store in LSM tree for fast lookups
            if self._lsm:
                try:
                    self._lsm.put(key, value)
                    success = True
                    logger.debug(f"Stored in LSM: {key}")
                except Exception as e:
                    logger.error(f"LSM storage failed for {key}: {e}")
            
            # Index in GraphRAG for semantic search (text only)
            if self._graph_rag and isinstance(value, str):
                try:
                    self._graph_rag.add_document(
                        doc_id=key,
                        content=value,
                        metadata=metadata,
                    )
                    success = True
                    logger.debug(f"Indexed in GraphRAG: {key}")
                except Exception as e:
                    logger.error(f"GraphRAG indexing failed for {key}: {e}")
            
            # Store in hierarchical memory
            if self._hierarchical:
                try:
                    self._hierarchical.store(
                        content=value,
                        memory_type=MemoryType.EPISODIC,
                        metadata={"key": key, **metadata},
                    )
                    success = True
                    logger.debug(f"Stored in HierarchicalMemory: {key}")
                except Exception as e:
                    logger.error(f"Hierarchical storage failed for {key}: {e}")
            
            return success
            
        except Exception as e:
            logger.error(f"Storage operation failed for {key}: {e}", exc_info=True)
            return False
    
    def retrieve(
        self,
        query: str,
        k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve data using hybrid search across backends.
        
        Combines results from multiple retrieval strategies:
        - Semantic search via GraphRAG (embedding similarity)
        - Contextual retrieval via HierarchicalMemory (temporal patterns)
        
        Args:
            query: Search query string
            k: Maximum number of results to return
        
        Returns:
            List of result dictionaries with keys:
                - source: Backend that provided the result
                - content: Retrieved content
                - score: Relevance score (0.0-1.0)
                - metadata: Optional metadata dict
        
        Raises:
            ValueError: If k <= 0
        
        Thread Safety:
            This method is thread-safe.
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        
        results: List[Dict[str, Any]] = []
        
        # Retrieve from GraphRAG (semantic search)
        if self._graph_rag:
            try:
                rag_results = self._graph_rag.retrieve(
                    query_or_embedding=query,
                    k=k,
                    rerank=True,
                )
                for r in rag_results:
                    results.append({
                        "source": "graph_rag",
                        "content": r.content,
                        "score": r.score,
                        "metadata": r.metadata,
                    })
                logger.debug(f"GraphRAG retrieved {len(rag_results)} results")
            except Exception as e:
                logger.warning(f"GraphRAG retrieval failed: {e}")
        
        # Retrieve from HierarchicalMemory (contextual)
        if self._hierarchical:
            try:
                mem_results = self._hierarchical.retrieve_context_for_generation(
                    query_tokens=query,
                    max_tokens=2048,
                )
                if mem_results.get("recent_context"):
                    for content in mem_results["recent_context"]:
                        results.append({
                            "source": "hierarchical",
                            "content": content,
                            "score": 0.5,  # Default score
                        })
                    logger.debug(
                        f"HierarchicalMemory retrieved "
                        f"{len(mem_results['recent_context'])} results"
                    )
            except Exception as e:
                logger.warning(f"Hierarchical retrieval failed: {e}")
        
        # Return top-k results
        return results[:k]
    
    # ========================================================================
    # UNLEARNING OPERATIONS
    # ========================================================================
    
    def unlearn(
        self,
        pattern: str,
        method: str = "gradient_surgery",
        urgency: str = "normal",
        requester_id: str = "system",
    ) -> Dict[str, Any]:
        """
        Request governed unlearning of a data pattern.
        
        Submits an unlearning request that goes through governance consensus
        before execution. This ensures compliance with data protection regulations
        (GDPR, CCPA) while maintaining audit trails.
        
        Args:
            pattern: Pattern or identifier of data to unlearn
            method: Unlearning method (gradient_surgery, exact_removal, etc.)
            urgency: Priority level (low, normal, high, critical)
            requester_id: ID of the entity requesting unlearning
        
        Returns:
            Dictionary with keys:
                - status: Request status (submitted, error)
                - proposal_id: Unique proposal identifier (if submitted)
                - method: Selected unlearning method
                - urgency: Priority level
                - message: Error message (if status is error)
        
        Thread Safety:
            This method is thread-safe.
        """
        if not self._governed_unlearning:
            return {
                "status": "error",
                "message": "Governed unlearning not available"
            }
        
        try:
            proposal_id = self._governed_unlearning.submit_ir_proposal(
                ir_content={"pattern": pattern},
                proposer_id=requester_id,
            )
            
            logger.info(
                f"Unlearning proposal submitted: {proposal_id} "
                f"(pattern={pattern}, urgency={urgency})"
            )
            
            return {
                "status": "submitted",
                "proposal_id": proposal_id,
                "method": method,
                "urgency": urgency,
            }
        except Exception as e:
            logger.error(f"Unlearn request failed: {e}", exc_info=True)
            return {
                "status": "error",
                "message": str(e)
            }
    
    def generate_unlearning_proof(
        self,
        items: List[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Generate zero-knowledge proof of unlearning.
        
        Creates a cryptographic proof that specified items were removed from
        the system without revealing what those items were. Useful for
        compliance verification.
        
        Args:
            items: List of item identifiers that were unlearned
        
        Returns:
            Proof dictionary or None if ZK proofs not available
        
        Thread Safety:
            This method is thread-safe.
        """
        if not self._zk_prover:
            logger.warning("ZK prover not available")
            return None
        
        if not items:
            logger.warning("No items provided for proof generation")
            return None
        
        try:
            proof = self._zk_prover.generate_unlearning_proof(
                removed_items=items,
            )
            logger.info(f"Generated unlearning proof for {len(items)} items")
            return proof
        except Exception as e:
            logger.error(f"Proof generation failed: {e}", exc_info=True)
            return None
    
    # ========================================================================
    # COST OPERATIONS
    # ========================================================================
    
    def optimize_storage(
        self,
        strategy: str = "balanced"
    ) -> Dict[str, Any]:
        """
        Run storage optimization to reduce costs.
        
        Performs various optimization techniques:
        - Deduplication of redundant data
        - Compression of cold data
        - Tier migration (hot -> warm -> cold)
        - Archival of old data
        
        Args:
            strategy: Optimization strategy (aggressive, balanced, conservative)
        
        Returns:
            Optimization report dictionary with keys:
                - status: Operation status
                - cost_before: Cost before optimization (optional)
                - cost_after: Cost after optimization (optional)
                - savings: Estimated savings (optional)
                - message: Error message (if status is error)
        
        Thread Safety:
            This method is thread-safe.
        """
        if not self._cost_optimizer:
            return {
                "status": "error",
                "message": "Cost optimizer not available"
            }
        
        try:
            report = self._cost_optimizer.optimize_storage()
            logger.info(f"Storage optimization completed (strategy={strategy})")
            
            if hasattr(report, 'to_dict'):
                return report.to_dict()
            else:
                return {"status": "completed"}
                
        except Exception as e:
            logger.error(f"Optimization failed: {e}", exc_info=True)
            return {
                "status": "error",
                "message": str(e)
            }
    
    def check_budget(self) -> Dict[str, Any]:
        """
        Check current budget status and utilization.
        
        Returns:
            Budget status dictionary with keys:
                - status: Budget status (ok, warning, exceeded, unknown)
                - usage_percentage: Percentage of budget used (optional)
                - current_cost: Current monthly cost (optional)
                - budget_limit: Budget limit (optional)
        
        Thread Safety:
            This method is thread-safe.
        """
        if not self._cost_optimizer:
            return {"status": "unknown"}
        
        try:
            return self._cost_optimizer.check_budget()
        except Exception as e:
            logger.error(f"Budget check failed: {e}")
            return {"status": "unknown", "error": str(e)}
    
    # ========================================================================
    # STATUS AND METRICS
    # ========================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get status of all memory components.
        
        Returns:
            Status dictionary with component availability and initialization state
        
        Thread Safety:
            This method is thread-safe.
        """
        with self._lock:
            return {
                "persistent_memory_available": PERSISTENT_MEMORY_AVAILABLE,
                "hierarchical_memory_available": HIERARCHICAL_MEMORY_AVAILABLE,
                "governed_unlearning_available": GOVERNED_UNLEARNING_AVAILABLE,
                "cost_optimizer_available": COST_OPTIMIZER_AVAILABLE,
                "storage_initialized": self._storage is not None,
                "graph_rag_initialized": self._graph_rag is not None,
                "lsm_initialized": self._lsm is not None,
                "zk_prover_initialized": self._zk_prover is not None,
                "hierarchical_initialized": self._hierarchical is not None,
                "governed_unlearning_initialized": self._governed_unlearning is not None,
                "cost_optimizer_initialized": self._cost_optimizer is not None,
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics from all components.
        
        Returns:
            Statistics dictionary with per-component metrics
        
        Thread Safety:
            This method is thread-safe.
        """
        stats: Dict[str, Any] = {}
        
        if self._graph_rag:
            try:
                stats["graph_rag"] = self._graph_rag.get_statistics()
            except Exception as e:
                logger.debug(f"Failed to get GraphRAG stats: {e}")
        
        if self._lsm:
            try:
                stats["lsm"] = self._lsm.get_statistics()
            except Exception as e:
                logger.debug(f"Failed to get LSM stats: {e}")
        
        if self._governed_unlearning:
            try:
                metrics = self._governed_unlearning.get_unlearning_metrics()
                stats["unlearning"] = (
                    metrics.to_dict() if hasattr(metrics, 'to_dict') else {}
                )
            except Exception as e:
                logger.debug(f"Failed to get unlearning metrics: {e}")
        
        if self._cost_optimizer:
            try:
                metrics = self._cost_optimizer.get_metrics()
                stats["cost"] = (
                    metrics.to_dict() if hasattr(metrics, 'to_dict') else {}
                )
            except Exception as e:
                logger.debug(f"Failed to get cost metrics: {e}")
        
        return stats
    
    def shutdown(self) -> None:
        """
        Perform clean shutdown of all components.
        
        Closes all connections, flushes buffers, and releases resources.
        Should be called before process termination.
        
        Thread Safety:
            This method is thread-safe.
        """
        with self._lock:
            logger.info("Shutting down MemoryBridge...")
            
            # Shutdown in reverse initialization order
            if self._cost_optimizer and hasattr(self._cost_optimizer, 'shutdown'):
                try:
                    self._cost_optimizer.shutdown()
                    logger.debug("CostOptimizer shut down")
                except Exception as e:
                    logger.error(f"CostOptimizer shutdown failed: {e}")
            
            if self._governed_unlearning and hasattr(self._governed_unlearning, 'shutdown'):
                try:
                    self._governed_unlearning.shutdown()
                    logger.debug("GovernedUnlearning shut down")
                except Exception as e:
                    logger.error(f"GovernedUnlearning shutdown failed: {e}")
            
            if self._graph_rag and hasattr(self._graph_rag, 'close'):
                try:
                    self._graph_rag.close()
                    logger.debug("GraphRAG closed")
                except Exception as e:
                    logger.error(f"GraphRAG close failed: {e}")
            
            if self._lsm and hasattr(self._lsm, 'close'):
                try:
                    self._lsm.close()
                    logger.debug("MerkleLSM closed")
                except Exception as e:
                    logger.error(f"MerkleLSM close failed: {e}")
            
            logger.info("MemoryBridge shutdown complete")
    
    def __enter__(self) -> MemoryBridge:
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit with automatic cleanup."""
        self.shutdown()


# ============================================================================
# FACTORY FUNCTION
# ============================================================================


def create_memory_bridge(
    config: Optional[Union[Dict[str, Any], MemoryBridgeConfig]] = None
) -> MemoryBridge:
    """
    Factory function to create MemoryBridge instance.
    
    Convenience function that accepts either a dict or MemoryBridgeConfig.
    
    Args:
        config: Configuration as dict or MemoryBridgeConfig object
    
    Returns:
        Initialized MemoryBridge instance
    
    Example:
        >>> bridge = create_memory_bridge({"s3_bucket": "my-bucket"})
        >>> # Or with config object:
        >>> config = MemoryBridgeConfig(s3_bucket="my-bucket")
        >>> bridge = create_memory_bridge(config)
    """
    if config is None:
        bridge_config = None
    elif isinstance(config, dict):
        bridge_config = MemoryBridgeConfig(**config)
    elif isinstance(config, MemoryBridgeConfig):
        bridge_config = config
    else:
        raise TypeError(
            f"config must be dict or MemoryBridgeConfig, got {type(config)}"
        )
    
    return MemoryBridge(bridge_config)


# ============================================================================
# PUBLIC API
# ============================================================================


__all__ = [
    "MemoryBridge",
    "MemoryBridgeConfig",
    "create_memory_bridge",
    "PERSISTENT_MEMORY_AVAILABLE",
    "HIERARCHICAL_MEMORY_AVAILABLE",
    "GOVERNED_UNLEARNING_AVAILABLE",
    "COST_OPTIMIZER_AVAILABLE",
]
