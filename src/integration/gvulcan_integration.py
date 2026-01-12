"""
Unified Integration Module for gvulcan Components with VULCAN System

Provides single entry point for all gvulcan integrations:
- DQS (Data Quality Score)
- OPA (Policy Enforcement)
- Merkle Trees (Data Integrity)
- Bloom Filters (Membership Testing)
- ZK Proofs (Zero-Knowledge Verification)
- Configuration (Unified Config)
- Vector Storage (Milvus Integration)
- CRC32C (Checksums)

Industry standard implementation with:
- Lazy initialization for efficient resource usage
- Comprehensive error handling
- Graceful degradation when components unavailable
- Thread-safe operations
- Detailed logging for observability
- Type safety and documentation
"""

import logging
from typing import Any, Dict, List, Optional
import threading

logger = logging.getLogger(__name__)

# Track availability of each component
GVULCAN_COMPONENTS = {
    "dqs": False,
    "opa": False,
    "merkle": False,
    "bloom": False,
    "zk": False,
    "config": False,
    "vector": False,
    "crc32c": False,
}

# Import all gvulcan components with graceful fallback
try:
    from gvulcan import (
        DQSScorer, DQSTracker, DQSComponents, DQSResult, compute_dqs,
        BloomFilter, CountingBloomFilter, ScalableBloomFilter,
        OPAClient, WriteBarrierInput, WriteBarrierResult, PolicyRegistry,
        MerkleTree, MerkleLSMDAG, MerkleProof,
        GVulcanConfig, ConfigurationManager, get_config,
        zk, merkle,
    )
    GVULCAN_COMPONENTS["dqs"] = True
    GVULCAN_COMPONENTS["opa"] = True
    GVULCAN_COMPONENTS["merkle"] = True
    GVULCAN_COMPONENTS["bloom"] = True
    GVULCAN_COMPONENTS["zk"] = True
    GVULCAN_COMPONENTS["config"] = True
    GVULCAN_AVAILABLE = True
    logger.info("gvulcan core modules loaded successfully")
except ImportError as e:
    logger.warning(f"gvulcan not fully available: {e}")
    GVULCAN_AVAILABLE = False
    # Set placeholders for graceful degradation
    DQSScorer = DQSTracker = DQSComponents = DQSResult = compute_dqs = None
    BloomFilter = CountingBloomFilter = ScalableBloomFilter = None
    OPAClient = WriteBarrierInput = WriteBarrierResult = PolicyRegistry = None
    MerkleTree = MerkleLSMDAG = MerkleProof = None
    GVulcanConfig = ConfigurationManager = get_config = None
    zk = merkle = None

# Try to import optional vector module
try:
    from gvulcan.vector import MilvusIndex, bootstrap_all_collections
    GVULCAN_COMPONENTS["vector"] = True
    logger.info("gvulcan.vector module loaded successfully")
except ImportError:
    MilvusIndex = bootstrap_all_collections = None
    logger.debug("gvulcan.vector not available")

# Try to import optional crc32c module  
try:
    from gvulcan.crc32c import crc32c, crc32c_combine, StreamingCRC32C
    GVULCAN_COMPONENTS["crc32c"] = True
    logger.info("gvulcan.crc32c module loaded successfully")
except ImportError:
    crc32c = crc32c_combine = StreamingCRC32C = None
    logger.debug("gvulcan.crc32c not available")


def get_component_status() -> Dict[str, bool]:
    """
    Get availability status of all gvulcan components.
    
    Returns:
        Dictionary mapping component names to availability status
    """
    return GVULCAN_COMPONENTS.copy()


def is_component_available(component: str) -> bool:
    """
    Check if a specific gvulcan component is available.
    
    Args:
        component: Component name (e.g., 'dqs', 'opa', 'merkle')
        
    Returns:
        True if component is available, False otherwise
    """
    return GVULCAN_COMPONENTS.get(component, False)


class GVulcanIntegration:
    """
    Unified integration class for gvulcan components.
    
    Provides lazy initialization and centralized access to all gvulcan features.
    Thread-safe implementation suitable for multi-threaded applications.
    
    Industry standard features:
    - Lazy initialization of expensive resources
    - Thread-safe singleton instances per component
    - Comprehensive error handling
    - Resource cleanup support
    - Detailed logging
    
    Example:
        >>> integration = GVulcanIntegration()
        >>> if integration.dqs_scorer:
        ...     result = integration.validate_data_quality(0.8, 0.9, 0.95)
        >>> if integration.opa_client:
        ...     allowed = integration.check_write_permission(0.7, {})
    """
    
    def __init__(self, config: Optional["GVulcanConfig"] = None):
        """
        Initialize gvulcan integration.
        
        Args:
            config: Optional GVulcanConfig instance for component configuration
        """
        self._config = config
        self._dqs_scorer = None
        self._opa_client = None
        self._merkle_dag = None
        self._bloom_filter = None
        self._lock = threading.Lock()
        
        logger.debug("GVulcanIntegration instance created")
    
    @property
    def dqs_scorer(self) -> Optional["DQSScorer"]:
        """
        Get or create DQS scorer instance.
        
        Thread-safe lazy initialization.
        
        Returns:
            DQSScorer instance if available, None otherwise
        """
        if not GVULCAN_COMPONENTS["dqs"]:
            return None
        
        if self._dqs_scorer is None:
            with self._lock:
                # Double-check after acquiring lock
                if self._dqs_scorer is None:
                    try:
                        self._dqs_scorer = DQSScorer(
                            model="v2",
                            reject_below=0.3,
                            quarantine_below=0.4
                        )
                        logger.info("DQS scorer initialized")
                    except Exception as e:
                        logger.error(f"Failed to initialize DQS scorer: {e}")
                        return None
        
        return self._dqs_scorer
    
    @property
    def opa_client(self) -> Optional["OPAClient"]:
        """
        Get or create OPA client instance.
        
        Thread-safe lazy initialization with caching and TTL support.
        
        Returns:
            OPAClient instance if available, None otherwise
        """
        if not GVULCAN_COMPONENTS["opa"]:
            return None
        
        if self._opa_client is None:
            with self._lock:
                # Double-check after acquiring lock
                if self._opa_client is None:
                    try:
                        # Check config for OPA URL
                        opa_url = None
                        if self._config and hasattr(self._config, 'opa'):
                            opa_url = getattr(self._config.opa, 'url', None)
                        
                        self._opa_client = OPAClient(
                            bundle_version="1.0.0",
                            opa_url=opa_url,
                            enable_cache=True,
                            cache_ttl_seconds=300,  # 5 minute TTL
                            enable_audit=True
                        )
                        logger.info(f"OPA client initialized (remote={'yes' if opa_url else 'no'})")
                    except Exception as e:
                        logger.error(f"Failed to initialize OPA client: {e}")
                        return None
        
        return self._opa_client
    
    @property
    def merkle_dag(self) -> Optional["MerkleLSMDAG"]:
        """
        Get or create Merkle LSM-DAG instance.
        
        Thread-safe lazy initialization.
        
        Returns:
            MerkleLSMDAG instance if available, None otherwise
        """
        if not GVULCAN_COMPONENTS["merkle"]:
            return None
        
        if self._merkle_dag is None:
            with self._lock:
                # Double-check after acquiring lock
                if self._merkle_dag is None:
                    try:
                        from gvulcan.merkle import HashAlgorithm
                        self._merkle_dag = MerkleLSMDAG(algorithm=HashAlgorithm.SHA256)
                        logger.info("Merkle LSM-DAG initialized")
                    except Exception as e:
                        logger.error(f"Failed to initialize Merkle DAG: {e}")
                        return None
        
        return self._merkle_dag
    
    def create_bloom_filter(
        self,
        expected_items: int = 10000,
        false_positive_rate: float = 0.01
    ) -> Optional["BloomFilter"]:
        """
        Create an optimally-sized bloom filter.
        
        Industry standard implementation:
        - Optimal sizing based on expected items and FPR
        - Configurable false positive rate
        - Memory-efficient storage
        
        Args:
            expected_items: Expected number of items to add
            false_positive_rate: Target false positive rate (0.0-1.0)
            
        Returns:
            BloomFilter instance if available, None otherwise
            
        Raises:
            ValueError: If parameters are invalid
        """
        if not GVULCAN_COMPONENTS["bloom"]:
            logger.warning("Bloom filter component not available")
            return None
        
        if expected_items <= 0:
            raise ValueError("expected_items must be positive")
        
        if not (0.0 < false_positive_rate < 1.0):
            raise ValueError("false_positive_rate must be between 0 and 1")
        
        try:
            bloom_filter = BloomFilter.create_optimal(expected_items, false_positive_rate)
            logger.info(
                f"Created bloom filter: {expected_items} items, "
                f"FPR={false_positive_rate}"
            )
            return bloom_filter
        except Exception as e:
            logger.error(f"Failed to create bloom filter: {e}")
            return None
    
    def validate_data_quality(
        self,
        pii_confidence: float,
        graph_completeness: float,
        syntactic_completeness: float
    ) -> Optional[Dict[str, Any]]:
        """
        Validate data quality using DQS system.
        
        Industry standard validation with:
        - Input validation
        - Comprehensive scoring
        - Decision thresholds
        
        Args:
            pii_confidence: PII confidence score (0.0-1.0)
            graph_completeness: Graph completeness score (0.0-1.0)
            syntactic_completeness: Syntactic completeness score (0.0-1.0)
            
        Returns:
            Dictionary with DQS result if scorer available, None otherwise
            
        Raises:
            ValueError: If input parameters are invalid
        """
        if self.dqs_scorer is None:
            logger.warning("DQS scorer not available for data quality validation")
            return None
        
        # Validate inputs
        for name, value in [
            ("pii_confidence", pii_confidence),
            ("graph_completeness", graph_completeness),
            ("syntactic_completeness", syntactic_completeness),
        ]:
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be between 0.0 and 1.0")
        
        try:
            components = DQSComponents(
                pii_confidence=pii_confidence,
                graph_completeness=graph_completeness,
                syntactic_completeness=syntactic_completeness
            )
            result = self.dqs_scorer.score(components)
            
            logger.info(
                f"DQS validation: score={result.score:.3f}, decision={result.decision}"
            )
            
            return {
                "score": result.score,
                "decision": result.decision,
                "components": {
                    "pii_confidence": pii_confidence,
                    "graph_completeness": graph_completeness,
                    "syntactic_completeness": syntactic_completeness,
                }
            }
        except Exception as e:
            logger.error(f"Error during DQS validation: {e}", exc_info=True)
            raise
    
    def check_write_permission(
        self,
        dqs_score: float,
        pii_info: Dict[str, Any]
    ) -> bool:
        """
        Check if write operation is permitted based on policies.
        
        Industry standard policy enforcement:
        - DQS-based gating
        - PII detection integration
        - Cached policy evaluation
        
        Args:
            dqs_score: Data Quality Score (0.0-1.0)
            pii_info: PII detection results
            
        Returns:
            True if write is permitted, False otherwise (defaults to True if OPA unavailable)
            
        Raises:
            ValueError: If dqs_score is invalid
        """
        if self.opa_client is None:
            logger.warning("OPA client not available, defaulting to allow")
            return True  # Default allow if OPA not available
        
        if not (0.0 <= dqs_score <= 1.0):
            raise ValueError("dqs_score must be between 0.0 and 1.0")
        
        try:
            barrier_input = WriteBarrierInput(dqs=dqs_score, pii=pii_info)
            result = self.opa_client.evaluate_write_barrier(barrier_input)
            
            logger.info(
                f"Write barrier: dqs={dqs_score:.3f}, allow={result.allow}, "
                f"quarantine={result.quarantine}"
            )
            
            return result.allow
        except Exception as e:
            logger.error(f"Error during write permission check: {e}", exc_info=True)
            # Fail closed for security
            return False
    
    def compute_data_integrity_hash(self, data: bytes) -> Optional[bytes]:
        """
        Compute integrity hash using Merkle DAG.
        
        Industry standard implementation:
        - Cryptographic hash function
        - Incremental updates
        - Proof generation support
        
        Args:
            data: Data to hash
            
        Returns:
            Merkle root hash if DAG available, None otherwise
            
        Raises:
            ValueError: If data is invalid
        """
        if self.merkle_dag is None:
            logger.warning("Merkle DAG not available for integrity hashing")
            return None
        
        if not isinstance(data, bytes):
            raise ValueError("data must be bytes")
        
        if len(data) == 0:
            logger.warning("Computing hash of empty data")
        
        try:
            self.merkle_dag.append_leaf(data)
            root = self.merkle_dag.current_root()
            
            logger.debug(f"Computed Merkle root: {root.hex()[:16]}...")
            return root
        except Exception as e:
            logger.error(f"Error computing integrity hash: {e}", exc_info=True)
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about integration components.
        
        Returns:
            Dictionary with statistics for each initialized component
        """
        stats = {
            "components_available": GVULCAN_COMPONENTS.copy(),
            "components_initialized": {}
        }
        
        # DQS stats
        if self._dqs_scorer is not None:
            try:
                stats["components_initialized"]["dqs"] = True
            except Exception as e:
                logger.warning(f"Error getting DQS stats: {e}")
        
        # OPA stats
        if self._opa_client is not None:
            try:
                stats["components_initialized"]["opa"] = True
                stats["opa_stats"] = self._opa_client.get_statistics()
            except Exception as e:
                logger.warning(f"Error getting OPA stats: {e}")
        
        # Merkle DAG stats
        if self._merkle_dag is not None:
            try:
                stats["components_initialized"]["merkle"] = True
                stats["merkle_stats"] = self._merkle_dag.get_stats()
            except Exception as e:
                logger.warning(f"Error getting Merkle stats: {e}")
        
        return stats
    
    def cleanup(self) -> None:
        """
        Clean up resources and reset component instances.
        
        Should be called when integration is no longer needed.
        """
        with self._lock:
            # Clear OPA cache if available
            if self._opa_client is not None:
                try:
                    self._opa_client.clear_cache()
                except Exception as e:
                    logger.warning(f"Error clearing OPA cache: {e}")
            
            # Reset all instances
            self._dqs_scorer = None
            self._opa_client = None
            self._merkle_dag = None
            self._bloom_filter = None
        
        logger.info("GVulcanIntegration cleaned up")


# Global integration instance with thread-safe access
_integration: Optional[GVulcanIntegration] = None
_integration_lock = threading.Lock()


def get_integration(config: Optional["GVulcanConfig"] = None) -> GVulcanIntegration:
    """
    Get or create global gvulcan integration instance.
    
    Thread-safe singleton pattern implementation.
    
    Args:
        config: Optional configuration (only used on first call)
        
    Returns:
        Global GVulcanIntegration instance
    """
    global _integration
    
    if _integration is None:
        with _integration_lock:
            # Double-check after acquiring lock
            if _integration is None:
                _integration = GVulcanIntegration(config=config)
                logger.info("Global gvulcan integration initialized")
    
    return _integration


def reset_integration() -> None:
    """
    Reset global integration instance.
    
    Useful for testing or when reconfiguration is needed.
    Cleans up resources before reset.
    """
    global _integration
    
    with _integration_lock:
        if _integration is not None:
            _integration.cleanup()
            _integration = None
            logger.info("Global gvulcan integration reset")


__all__ = [
    "GVULCAN_AVAILABLE",
    "GVULCAN_COMPONENTS",
    "GVulcanIntegration",
    "get_component_status",
    "is_component_available",
    "get_integration",
    "reset_integration",
]
