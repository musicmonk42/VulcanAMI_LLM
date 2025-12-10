"""
Milvus Vector Index Client

This module provides a comprehensive client for multi-tier vector search with
support for different quantization methods, dynamic routing, and performance optimization.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pymilvus import Collection, connections, utility

logger = logging.getLogger(__name__)


class SearchTier(Enum):
    """Vector search quality tiers"""

    A_PLUS = "A+"  # Highest quality, FP16 + HNSW + Rotational 8-bit
    A = "A"  # High quality, FP16 + HNSW
    B = "B"  # Medium quality, Rotational 8-bit + IVF
    C = "C"  # Storage-optimized, INT4 + ECC + Disk


@dataclass
class TierConfig:
    """
    Configuration for a search tier.

    Attributes:
        name: Tier name
        recall_at_50: Expected recall @ 50 metric
        size_factor: Relative size compared to baseline
        format: Encoding format description
        collection_name: Milvus collection name
        index_type: Index type (HNSW, IVF_FLAT, etc.)
        metric_type: Distance metric (L2, IP, COSINE)
    """

    name: str
    recall_at_50: float
    size_factor: float
    format: str
    collection_name: str = ""
    index_type: str = "HNSW"
    metric_type: str = "L2"

    def __post_init__(self):
        if not self.collection_name:
            self.collection_name = (
                f"vectors_tier_{self.name.lower().replace('+', 'plus')}"
            )


@dataclass
class SearchResult:
    """
    Single search result.

    Attributes:
        id: Vector ID
        distance: Distance/similarity score
        metadata: Associated metadata
    """

    id: str
    distance: float
    metadata: Dict[str, Any]


@dataclass
class SearchResponse:
    """
    Response from vector search.

    Attributes:
        results: List of search results
        tier: Tier used for search
        latency_ms: Search latency in milliseconds
        total_results: Total number of results found
    """

    results: List[SearchResult]
    tier: str
    latency_ms: float
    total_results: int


class MilvusIndex:
    """
    Multi-tier vector index client for Milvus.

    Provides intelligent routing between quality tiers based on requirements,
    with support for fallback, batching, and comprehensive metrics.

    Example:
        index = MilvusIndex(uri="http://localhost:19530")
        index.connect()

        # Add vectors
        vectors = np.random.randn(100, 768)
        ids = [f"vec_{i}" for i in range(100)]
        index.add_vectors(vectors, ids)

        # Search
        query = np.random.randn(768)
        results = index.search(query, k=10, tier=SearchTier.A)
    """

    # Tier configurations
    TIERS = [
        TierConfig(
            name="A+",
            recall_at_50=0.989,
            size_factor=0.5,
            format="FP16+HNSW+Rot-8bit",
            index_type="HNSW",
            metric_type="L2",
        ),
        TierConfig(
            name="A",
            recall_at_50=0.982,
            size_factor=1.0,
            format="FP16+HNSW",
            index_type="HNSW",
            metric_type="L2",
        ),
        TierConfig(
            name="B",
            recall_at_50=0.975,
            size_factor=0.4,
            format="Rot-8bit+IVF",
            index_type="IVF_FLAT",
            metric_type="L2",
        ),
        TierConfig(
            name="C",
            recall_at_50=0.970,
            size_factor=0.2,
            format="INT4+ECC+Disk",
            index_type="IVF_PQ",
            metric_type="L2",
        ),
    ]

    def __init__(
        self,
        uri: str,
        default_tier: SearchTier = SearchTier.A,
        enable_fallback: bool = True,
        connection_timeout: float = 30.0,
    ):
        """
        Initialize Milvus index client.

        Args:
            uri: Milvus server URI
            default_tier: Default search tier
            enable_fallback: Whether to fallback to lower tiers on failure
            connection_timeout: Connection timeout in seconds
        """
        self.uri = uri
        self.default_tier = default_tier
        self.enable_fallback = enable_fallback
        self.connection_timeout = connection_timeout

        # Collections for each tier
        self.collections: Dict[str, Collection] = {}
        self.tier_configs = {tier.name: tier for tier in self.TIERS}

        # Statistics
        self.stats = {
            "total_searches": 0,
            "total_inserts": 0,
            "tier_usage": {tier.name: 0 for tier in self.TIERS},
            "fallback_count": 0,
        }

        self._connected = False

        logger.info(
            f"Initialized MilvusIndex: uri={uri}, default_tier={default_tier.value}"
        )

    def connect(self, alias: str = "default") -> None:
        """
        Connect to Milvus server.

        Args:
            alias: Connection alias
        """
        if self._connected:
            logger.warning("Already connected to Milvus")
            return

        try:
            host, port = self._parse_uri(self.uri)
            connections.connect(
                alias=alias, host=host, port=port, timeout=self.connection_timeout
            )
            self._connected = True
            logger.info(f"Connected to Milvus at {self.uri}")

            # Load collections
            self._load_collections()

        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise

    def _parse_uri(self, uri: str) -> Tuple[str, str]:
        """Parse URI into host and port"""
        uri = uri.replace("http://", "").replace("https://", "")

        if ":" in uri:
            host, port = uri.split(":", 1)
        else:
            host = uri
            port = "19530"

        return host, port

    def _load_collections(self) -> None:
        """Load existing collections for each tier"""
        for tier in self.TIERS:
            collection_name = tier.collection_name

            if utility.has_collection(collection_name):
                try:
                    collection = Collection(collection_name)
                    collection.load()
                    self.collections[tier.name] = collection
                    logger.info(
                        f"Loaded collection for tier {tier.name}: {collection_name}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to load collection {collection_name}: {e}")
            else:
                logger.info(
                    f"Collection {collection_name} does not exist for tier {tier.name}"
                )

    def add_vectors(
        self,
        vectors: np.ndarray,
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        tier: Optional[SearchTier] = None,
    ) -> Dict[str, int]:
        """
        Add vectors to the index.

        Args:
            vectors: Numpy array of vectors (N x D)
            ids: List of vector IDs
            metadata: Optional metadata for each vector
            tier: Specific tier to insert into (None = all tiers)

        Returns:
            Dictionary with insert counts per tier
        """
        if not self._connected:
            raise RuntimeError("Not connected to Milvus. Call connect() first.")

        if vectors.shape[0] != len(ids):
            raise ValueError("Number of vectors must match number of IDs")

        if metadata and len(metadata) != len(ids):
            raise ValueError("Number of metadata entries must match number of IDs")

        results = {}
        target_tiers = (
            [tier]
            if tier
            else [SearchTier.A_PLUS, SearchTier.A, SearchTier.B, SearchTier.C]
        )

        for search_tier in target_tiers:
            tier_config = self.tier_configs.get(search_tier.value)
            if not tier_config:
                continue

            collection = self.collections.get(search_tier.value)
            if not collection:
                logger.warning(
                    f"Collection not loaded for tier {search_tier.value}, skipping"
                )
                continue

            try:
                # Prepare data for insertion
                data = [ids, vectors.to[)]

                # Add metadata if provided
                if metadata:
                    # Extract metadata fields
                    for key in metadata[0].keys():
                        data.append([m.get(key) for m in metadata])

                # Insert
                insert_result = collection.insert(data)
                results[search_tier.value] = insert_result.insert_count

                self.stats["total_inserts"] += insert_result.insert_count

                logger.info(
                    f"Inserted {insert_result.insert_count} vectors into tier {search_tier.value}"
                )

            except Exception as e:
                logger.error(f"Failed to insert into tier {search_tier.value}: {e}")
                results[search_tier.value] = 0

        return results

    def search(
        self,
        query_vec: np.ndarray,
        k: int = 50,
        tier: Optional[SearchTier] = None,
        search_params: Optional[Dict[str, Any]] = None,
    ) -> SearchResponse:
        """
        Search for similar vectors.

        Args:
            query_vec: Query vector (D-dimensional)
            k: Number of results to return
            tier: Search tier (None = use default)
            search_params: Optional search parameters

        Returns:
            SearchResponse with results
        """
        if not self._connected:
            raise RuntimeError("Not connected to Milvus. Call connect() first.")

        search_tier = tier or self.default_tier
        time.time()

        # Try primary tier
        try:
            result = self._search_tier(query_vec, k, search_tier, search_params)
            return result

        except Exception as e:
            logger.error(f"Search failed on tier {search_tier.value}: {e}")

            # Try fallback
            if self.enable_fallback:
                for fallback_tier in self.TIERS:
                    if fallback_tier.name == search_tier.value:
                        continue

                    try:
                        logger.info(f"Attempting fallback to tier {fallback_tier.name}")
                        self.stats["fallback_count"] += 1
                        result = self._search_tier(
                            query_vec, k, SearchTier(fallback_tier.name), search_params
                        )
                        return result
                    except Exception as fallback_e:
                        logger.error(
                            f"Fallback to {fallback_tier.name} failed: {fallback_e}"
                        )

            # All attempts failed
            raise RuntimeError(f"Search failed on all tiers: {e}")

    def _search_tier(
        self,
        query_vec: np.ndarray,
        k: int,
        tier: SearchTier,
        search_params: Optional[Dict[str, Any]],
    ) -> SearchResponse:
        """Execute search on a specific tier"""
        collection = self.collections.get(tier.value)
        if not collection:
            raise ValueError(f"Collection not available for tier {tier.value}")

        # Prepare search params
        if search_params is None:
            tier_config = self.tier_configs[tier.value]
            if tier_config.index_type == "HNSW":
                search_params = {"metric_type": "L2", "params": {"ef": 64}}
            elif tier_config.index_type == "IVF_FLAT":
                search_params = {"metric_type": "L2", "params": {"nprobe": 16}}
            else:
                search_params = {"metric_type": "L2", "params": {}}

        start_time = time.time()

        # Execute search
        search_results = collection.search(
            data=[query_vec.to[)],
            anns_field="vector",
            param=search_params,
            limit=k,
            output_fields=["*"],
        )

        latency_ms = (time.time() - start_time) * 1000

        # Parse results
        results = []
        for hits in search_results:
            for hit in hits:
                result = SearchResult(
                    id=str(hit.id),
                    distance=float(hit.distance),
                    metadata=hit.entity.fields if hasattr(hit, "entity") else {},
                )
                results.append(result)

        # Update stats
        self.stats["total_searches"] += 1
        self.stats["tier_usage"][tier.value] = (
            self.stats["tier_usage"].get(tier.value, 0) + 1
        )

        logger.debug(
            f"Search on tier {tier.value}: {len(results)} results in {latency_ms:.2f}ms"
        )

        return SearchResponse(
            results=results,
            tier=tier.value,
            latency_ms=latency_ms,
            total_results=len(results),
        )

    def batch_search(
        self, query_vecs: np.ndarray, k: int = 50, tier: Optional[SearchTier] = None
    ) -> List[SearchResponse]:
        """
        Search for multiple query vectors.

        Args:
            query_vecs: Array of query vectors (N x D)
            k: Number of results per query
            tier: Search tier

        Returns:
            List of SearchResponse objects
        """
        responses = []

        for i, query_vec in enumerate(query_vecs):
            try:
                response = self.search(query_vec, k=k, tier=tier)
                responses.append(response)
            except Exception as e:
                logger.error(f"Batch search failed for query {i}: {e}")
                # Add empty response
                responses.append(
                    SearchResponse(
                        results=[],
                        tier=tier.value if tier else self.default_tier.value,
                        latency_ms=0.0,
                        total_results=0,
                    )
                )

        return responses

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        return {
            "connected": self._connected,
            "total_searches": self.stats["total_searches"],
            "total_inserts": self.stats["total_inserts"],
            "tier_usage": self.stats["tier_usage"],
            "fallback_count": self.stats["fallback_count"],
            "collections_loaded": len(self.collections),
            "available_tiers": list(self.collections.keys()),
        }

    def disconnect(self) -> None:
        """Disconnect from Milvus"""
        if self._connected:
            connections.disconnect("default")
            self._connected = False
            logger.info("Disconnected from Milvus")

    def __del__(self):
        """Cleanup on deletion"""
        if self._connected:
            try:
                self.disconnect()
            except Exception:
                pass


def create_index(uri: str, **kwargs) -> MilvusIndex:
    """
    Convenience function to create a Milvus index.

    Args:
        uri: Milvus server URI
        **kwargs: Additional arguments for MilvusIndex

    Returns:
        MilvusIndex instance
    """
    return MilvusIndex(uri=uri, **kwargs)
