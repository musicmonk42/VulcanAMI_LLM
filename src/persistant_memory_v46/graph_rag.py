"""
graph_rag.py – Production-Ready GraphRAG for Vulcan Persistent Memory v46

Implements core enhancement categories with proper error handling and fallbacks.
"""

import asyncio
import json
import logging
import os
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Optional imports with fallbacks
try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available, using numpy-based search")

try:
    import networkx as nx

    NX_AVAILABLE = True
except ImportError:
    NX_AVAILABLE = False
    logger.warning("NetworkX not available, using basic graph")

try:
    from rank_bm25 import BM25Okapi

    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.warning("BM25 not available, using TF-IDF fallback")

try:
    from sentence_transformers import CrossEncoder, SentenceTransformer

    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    logger.warning("Sentence-transformers not available, using mock embeddings")

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


@dataclass
class GraphNode:
    node_id: str
    content: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    neighbors: Set[str] = field(default_factory=set)
    modality: str = "text"  # text, image, audio
    timestamp: float = field(default_factory=time.time)


@dataclass
class RetrievalResult:
    node_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "graph"
    hop: int = 0


class SimpleGraph:
    """Fallback graph implementation when NetworkX is not available."""

    def __init__(self):
        self.nodes = {}
        self.edges = defaultdict(set)
        self.edge_attrs = {}

    def add_node(self, node_id, **attrs):
        self.nodes[node_id] = attrs

    def add_edge(self, src, dst, **attrs):
        self.edges[src].add(dst)
        self.edge_attrs[(src, dst)] = attrs

    def remove_node(self, node_id):
        if node_id in self.nodes:
            del self.nodes[node_id]
        if node_id in self.edges:
            del self.edges[node_id]
        for src in list(self.edges.keys()):
            self.edges[src].discard(node_id)

    def neighbors(self, node_id):
        return self.edges.get(node_id, set())


class LRUCache:
    """Simple LRU cache implementation."""

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.cache = {}
        self.order = deque()

    def get(self, key: str):
        if key not in self.cache:
            return None
        self.order.remove(key)
        self.order.append(key)
        return self.cache[key]

    def put(self, key: str, value: Any):
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) >= self.capacity:
            oldest = self.order.popleft()
            del self.cache[oldest]
        self.cache[key] = value
        self.order.append(key)


class GraphRAG:
    """Production-ready GraphRAG with graceful degradation."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        embedding_dim: int = 384,
        use_faiss: bool = True,
        cache_capacity: int = 1000,
        **kwargs,
    ):
        self.config = self._load_config(config_path)
        self.embedding_dim = embedding_dim
        self.use_faiss = use_faiss and FAISS_AVAILABLE

        # Initialize embedding model
        if ST_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
                self.cross_encoder = CrossEncoder(cross_encoder_model)
            except Exception as e:
                logger.warning(f"Failed to load models: {e}, using mock embeddings")
                self.embedding_model = None
                self.cross_encoder = None
        else:
            self.embedding_model = None
            self.cross_encoder = None

        # Initialize vector index
        if self.use_faiss:
            # Use simple Flat index to support all operations
            self.vector_index = faiss.IndexFlatL2(embedding_dim)
        else:
            self.vector_index = None
            self.embeddings_list = []

        self.index_to_node_id: Dict[int, str] = {}
        self.node_id_to_index: Dict[str, int] = {}

        # Initialize graph
        self.graph = nx.DiGraph() if NX_AVAILABLE else SimpleGraph()

        # Core data structures
        self.nodes: Dict[str, GraphNode] = {}
        self.bm25_index = None
        self.bm25_corpus = []
        self.bm25_doc_ids = []

        # Caching
        self.query_cache = LRUCache(cache_capacity)

        # Statistics
        self.stats = {
            "total_nodes": 0,
            "total_edges": 0,
            "queries": 0,
            "cache_hits": 0,
            "rerank_calls": 0,
        }

        logger.info(
            f"GraphRAG initialized (FAISS: {self.use_faiss}, NX: {NX_AVAILABLE})"
        )

    def _load_config(self, path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not path or not os.path.exists(path):
            return {}

        if YAML_AVAILABLE:
            try:
                with open(path, encoding="utf-8") as f:
                    return yaml.safe_load(f)
            except Exception as e:
                logger.error(f"Failed to load YAML config {path}: {e}")
                return {}
        else:
            # Fallback to JSON
            if path.endswith(".json"):
                try:
                    with open(path, encoding="utf-8") as f:
                        return json.load(f)
                except Exception as e:
                    logger.error(f"Failed to load JSON config {path}: {e}")
                    return {}
            return {}

    def _get_text_embedding(self, text: str) -> np.ndarray:
        """Generate text embedding."""
        if self.embedding_model:
            return self.embedding_model.encode(
                text, convert_to_numpy=True, normalize_embeddings=True
            )
        else:
            # Mock embedding using hash
            np.random.seed(hash(text) % (2**32 - 1))
            emb = np.random.randn(self.embedding_dim)
            return emb / np.linalg.norm(emb)

    def _semantic_chunk(self, text: str, max_tokens: int = 256) -> List[str]:
        """Split text into semantic chunks."""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        chunks = []
        current = ""

        for s in sentences:
            # Simple word count instead of tokenization
            if len(current.split()) + len(s.split()) <= max_tokens:
                current += " " + s
            else:
                if current:
                    chunks.append(current.strip())
                current = s

        if current:
            chunks.append(current.strip())

        return chunks if chunks else [text]

    def add_document(
        self,
        doc_id: str,
        content: Union[str, bytes],
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[np.ndarray] = None,
        chunks: Optional[List[Tuple[str, str, np.ndarray]]] = None,
        auto_chunk: bool = True,
        **kwargs,
    ) -> None:
        """Add document with automatic chunking and indexing."""
        if doc_id in self.nodes:
            logger.warning(f"Document {doc_id} exists. Overwriting.")
            self._remove_node(doc_id)

        metadata = metadata or {}
        content_str = (
            content.decode("utf-8", errors="ignore")
            if isinstance(content, bytes)
            else content
        )

        # Create parent node
        parent_embedding = (
            embedding
            if embedding is not None
            else self._get_text_embedding(content_str[:1000])
        )
        parent_node = GraphNode(
            node_id=doc_id,
            content=content_str[:1000],  # Store summary in parent
            embedding=parent_embedding,
            metadata=metadata,
        )
        self.nodes[doc_id] = parent_node
        self.graph.add_node(doc_id, **metadata)

        # Handle chunks
        if chunks:
            # Use provided chunks
            chunk_list = chunks
        elif auto_chunk and len(content_str) > 100:
            # Auto-chunk
            chunk_texts = self._semantic_chunk(content_str)
            chunk_list = [
                (f"{doc_id}_chunk_{i}", text, self._get_text_embedding(text))
                for i, text in enumerate(chunk_texts)
            ]
        else:
            # Single chunk
            chunk_list = [(f"{doc_id}_chunk_0", content_str, parent_embedding)]

        # Add chunks
        for chunk_id, chunk_text, chunk_emb in chunk_list:
            chunk_metadata = {**metadata, "parent_doc_id": doc_id}

            chunk_node = GraphNode(
                node_id=chunk_id,
                content=chunk_text,
                embedding=chunk_emb,
                metadata=chunk_metadata,
            )
            self.nodes[chunk_id] = chunk_node
            self.graph.add_node(chunk_id, **chunk_metadata)

            # Add to vector index
            idx = len(self.index_to_node_id)
            if self.use_faiss:
                try:
                    self.vector_index.add(chunk_emb.reshape(1, -1))
                except Exception as e:
                    logger.error(
                        f"Failed adding embedding to FAISS index for {chunk_id}: {e}"
                    )
            else:
                self.embeddings_list.append(chunk_emb)

            self.index_to_node_id[idx] = chunk_id
            self.node_id_to_index[chunk_id] = idx

            # Link to parent
            self.graph.add_edge(doc_id, chunk_id, type="has_chunk")
            parent_node.neighbors.add(chunk_id)

            # Update BM25 corpus
            self.bm25_corpus.append(chunk_text)
            self.bm25_doc_ids.append(chunk_id)

        # Rebuild BM25 index
        if BM25_AVAILABLE and self.bm25_corpus:
            try:
                tokenized = [text.lower().split() for text in self.bm25_corpus]
                self.bm25_index = BM25Okapi(tokenized)
            except Exception as e:
                logger.error(f"Failed to build BM25 index: {e}")

        self.stats["total_nodes"] = len(self.nodes)
        self.stats["total_edges"] += len(chunk_list)

        logger.debug(f"Added document {doc_id} with {len(chunk_list)} chunks")

    def _remove_node(self, node_id: str):
        """Remove a node from all indices."""
        if node_id not in self.nodes:
            return

        # Remove from graph
        try:
            self.graph.remove_node(node_id)
        except Exception as e:
            logger.error(f"Failed to remove node {node_id} from graph: {e}")

        # Remove from vector index if it's a chunk (mark logical deletion)
        if node_id in self.node_id_to_index:
            idx = self.node_id_to_index[node_id]
            del self.node_id_to_index[node_id]
            if idx in self.index_to_node_id:
                del self.index_to_node_id[idx]

        # Remove from nodes
        del self.nodes[node_id]

    def add_edge(
        self,
        src: str,
        dst: str,
        rel_type: str = "related",
        metadata: Optional[Dict] = None,
    ):
        """Add edge between nodes."""
        if src not in self.nodes or dst not in self.nodes:
            logger.warning(f"Cannot add edge: {src} or {dst} not found")
            return

        try:
            self.graph.add_edge(src, dst, type=rel_type, **(metadata or {}))
            self.nodes[src].neighbors.add(dst)
            self.stats["total_edges"] += 1
        except Exception as e:
            logger.error(f"Failed to add edge {src}->{dst}: {e}")

    def retrieve(
        self,
        query: Union[str, np.ndarray],
        k: int = 10,
        use_rerank: bool = True,
        use_hybrid: bool = True,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[RetrievalResult]:
        """Retrieve relevant documents."""
        self.stats["queries"] += 1

        # Check cache
        if isinstance(query, str):
            cache_key = f"{query}|{k}|{use_rerank}|{use_hybrid}"
            cached = self.query_cache.get(cache_key)
            if cached:
                self.stats["cache_hits"] += 1
                return cached
        else:
            cache_key = None

        # Get query embedding
        if isinstance(query, np.ndarray):
            query_emb = query
            query_text = ""
        else:
            query_emb = self._get_text_embedding(query)
            query_text = query

        # Vector search
        results = self._vector_search(query_emb, k * 2)

        # Hybrid search with BM25
        if use_hybrid and query_text and BM25_AVAILABLE and self.bm25_index:
            bm25_results = self._bm25_search(query_text, k)
            results = self._merge_results(results, bm25_results)

        # Apply filters
        if filters:
            results = self._apply_filters(results, filters)

        # Rerank
        if use_rerank and query_text and self.cross_encoder:
            results = self._rerank(query_text, results[: k * 2])

        # Take top k
        results = results[:k]

        # Update cache
        if cache_key:
            self.query_cache.put(cache_key, results)

        return results

    def _vector_search(self, query_emb: np.ndarray, k: int) -> List[RetrievalResult]:
        """Perform vector similarity search."""
        results = []

        if self.use_faiss and self.vector_index and self.vector_index.ntotal > 0:
            try:
                k_search = min(k, self.vector_index.ntotal)
                D, I = self.vector_index.search(query_emb.reshape(1, -1), k_search)

                for dist, idx in zip(D[0], I[0]):
                    if idx in self.index_to_node_id:
                        node_id = self.index_to_node_id[idx]
                        if node_id in self.nodes:
                            node = self.nodes[node_id]
                            score = 1.0 / (1.0 + dist)  # Convert distance to similarity
                            results.append(
                                RetrievalResult(
                                    node_id=node_id,
                                    content=node.content,
                                    score=float(score),
                                    metadata=node.metadata,
                                    source="vector",
                                )
                            )
            except Exception as e:
                logger.error(f"FAISS search failed, falling back to numpy search: {e}")
                return self._vector_search_numpy(query_emb, k)
        else:
            results = self._vector_search_numpy(query_emb, k)

        return results

    def _vector_search_numpy(
        self, query_emb: np.ndarray, k: int
    ) -> List[RetrievalResult]:
        """Numpy-based similarity search fallback."""
        scores = []
        for idx, emb in enumerate(self.embeddings_list):
            if idx in self.index_to_node_id:
                sim = np.dot(query_emb, emb) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-10
                )
                scores.append((idx, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in scores[:k]:
            if idx in self.index_to_node_id:
                node_id = self.index_to_node_id[idx]
                if node_id in self.nodes:
                    node = self.nodes[node_id]
                    results.append(
                        RetrievalResult(
                            node_id=node_id,
                            content=node.content,
                            score=float(score),
                            metadata=node.metadata,
                            source="vector",
                        )
                    )
        return results

    def _bm25_search(self, query: str, k: int) -> List[RetrievalResult]:
        """BM25 keyword search."""
        if not self.bm25_index:
            return []

        tokenized_query = query.lower().split()
        try:
            scores = self.bm25_index.get_scores(tokenized_query)
        except Exception as e:
            logger.error(f"BM25 scoring failed: {e}")
            return []

        # Get top k
        top_indices = np.argsort(scores)[-k:][::-1]

        results = []
        for idx in top_indices:
            if idx < len(self.bm25_doc_ids):
                node_id = self.bm25_doc_ids[idx]
                if node_id in self.nodes:
                    node = self.nodes[node_id]
                    results.append(
                        RetrievalResult(
                            node_id=node_id,
                            content=node.content,
                            score=float(scores[idx]),
                            metadata=node.metadata,
                            source="bm25",
                        )
                    )

        return results

    def _merge_results(
        self, results1: List[RetrievalResult], results2: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Merge and re-score results from different sources."""
        merged = {}

        # Normalize scores
        max_score1 = max([r.score for r in results1], default=1.0)
        max_score2 = max([r.score for r in results2], default=1.0)

        for r in results1:
            merged[r.node_id] = r.score / (max_score1 + 1e-10) * 0.7

        for r in results2:
            score = r.score / (max_score2 + 1e-10) * 0.3
            if r.node_id in merged:
                merged[r.node_id] += score
            else:
                merged[r.node_id] = score

        # Create merged results
        final_results = []
        for node_id, score in merged.items():
            if node_id in self.nodes:
                node = self.nodes[node_id]
                final_results.append(
                    RetrievalResult(
                        node_id=node_id,
                        content=node.content,
                        score=score,
                        metadata=node.metadata,
                        source="hybrid",
                    )
                )

        final_results.sort(key=lambda x: x.score, reverse=True)
        return final_results

    def _apply_filters(
        self, results: List[RetrievalResult], filters: Dict[str, Any]
    ) -> List[RetrievalResult]:
        """Apply metadata filters to results."""
        filtered = []

        for r in results:
            match = True
            for key, value in filters.items():
                if key not in r.metadata or r.metadata[key] != value:
                    match = False
                    break

            if match:
                filtered.append(r)

        return filtered

    def _rerank(
        self, query: str, results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Rerank results using cross-encoder."""
        if not results or not self.cross_encoder:
            return results

        self.stats["rerank_calls"] += 1

        try:
            pairs = [
                (query, r.content[:500]) for r in results
            ]  # Truncate for efficiency
            scores = self.cross_encoder.predict(pairs)

            for r, score in zip(results, scores):
                r.score = float(score)

            results.sort(key=lambda x: x.score, reverse=True)
        except Exception as e:
            logger.error(f"Cross-encoder rerank failed: {e}")

        return results

    async def retrieve_async(self, *args, **kwargs) -> List[RetrievalResult]:
        """Async version of retrieve."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.retrieve, *args, **kwargs)

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the GraphRAG system."""
        cache_hit_rate = self.stats["cache_hits"] / max(1, self.stats["queries"])

        stats = {
            **self.stats,
            "cache_hit_rate": cache_hit_rate,
            "index_size": self.vector_index.ntotal
            if (self.use_faiss and self.vector_index)
            else len(getattr(self, "embeddings_list", [))),
            "graph_nodes": len(self.nodes),
            "bm25_docs": len(self.bm25_corpus),
        }

        return stats

    def save(self, path: str):
        """Save the GraphRAG to disk."""
        os.makedirs(path, exist_ok=True)

        # Save vector index
        if self.use_faiss and self.vector_index:
            try:
                faiss.write_index(self.vector_index, os.path.join(path, "index.faiss"))
            except Exception as e:
                logger.error(f"Failed to save FAISS index: {e}")
        else:
            try:
                np.save(os.path.join(path, "embeddings.npy"), self.embeddings_list)
            except Exception as e:
                logger.error(f"Failed to save embeddings list: {e}")

        # Save metadata
        meta = {
            "index_to_node_id": self.index_to_node_id,
            "node_id_to_index": self.node_id_to_index,
            "bm25_corpus": self.bm25_corpus,
            "bm25_doc_ids": self.bm25_doc_ids,
            "stats": self.stats,
        }

        try:
            with open(os.path.join(path, "metadata.json", encoding="utf-8"), "w") as f:
                json.dump(meta, f)
        except Exception as e:
            logger.error(f"Failed to save metadata.json: {e}")

        # Save nodes
        nodes_data = {}
        for node_id, node in self.nodes.items():
            try:
                nodes_data[node_id] = {
                    "content": node.content,
                    "metadata": node.metadata,
                    "neighbors": list(node.neighbors),
                    "modality": node.modality,
                }
            except Exception as e:
                logger.error(f"Error serializing node {node_id}: {e}")

        try:
            with open(os.path.join(path, "nodes.json", encoding="utf-8"), "w") as f:
                json.dump(nodes_data, f)
        except Exception as e:
            logger.error(f"Failed to save nodes.json: {e}")

        logger.info(f"Saved GraphRAG to {path}")

    def load(self, path: str):
        """Load GraphRAG from disk."""
        # Load vector index
        if self.use_faiss and os.path.exists(os.path.join(path, "index.faiss")):
            try:
                self.vector_index = faiss.read_index(os.path.join(path, "index.faiss"))
            except Exception as e:
                logger.error(f"Failed to load FAISS index: {e}")
                self.vector_index = faiss.IndexFlatL2(self.embedding_dim)
        else:
            emb_path = os.path.join(path, "embeddings.npy")
            if os.path.exists(emb_path):
                try:
                    self.embeddings_list = np.load(emb_path, allow_pickle=True).tolist()
                except Exception as e:
                    logger.error(f"Failed to load embeddings list: {e}")
                    self.embeddings_list = []

        # Load metadata
        meta_path = os.path.join(path, "metadata.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
                    self.index_to_node_id = {
                        int(k): v for k, v in meta.get("index_to_node_id", {}).items()
                    }
                    self.node_id_to_index = meta.get("node_id_to_index", {})
                    self.bm25_corpus = meta.get("bm25_corpus", [])
                    self.bm25_doc_ids = meta.get("bm25_doc_ids", [])
                    self.stats.update(meta.get("stats", {}))
            except Exception as e:
                logger.error(f"Failed to load metadata.json: {e}")
        else:
            logger.warning("metadata.json not found during load")

        # Rebuild BM25 index
        if BM25_AVAILABLE and self.bm25_corpus:
            try:
                tokenized = [text.lower().split() for text in self.bm25_corpus]
                self.bm25_index = BM25Okapi(tokenized)
            except Exception as e:
                logger.error(f"Failed to rebuild BM25 index: {e}")

        # Load nodes
        nodes_path = os.path.join(path, "nodes.json")
        if os.path.exists(nodes_path):
            try:
                with open(nodes_path, encoding="utf-8") as f:
                    nodes_data = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load nodes.json: {e}")
                nodes_data = {}
        else:
            logger.warning("nodes.json not found during load")
            nodes_data = {}

        for node_id, data in nodes_data.items():
            try:
                node = GraphNode(
                    node_id=node_id,
                    content=data["content"],
                    metadata=data["metadata"],
                    neighbors=set(data["neighbors"]),
                    modality=data.get("modality", "text"),
                )
                self.nodes[node_id] = node

                # Rebuild graph
                self.graph.add_node(node_id, **data["metadata"])
                for neighbor in data["neighbors"]:
                    if neighbor in nodes_data:
                        self.graph.add_edge(node_id, neighbor)
            except Exception as e:
                logger.error(f"Failed recreating node {node_id}: {e}")

        logger.info(f"Loaded GraphRAG from {path}")

    def close(self):
        """Clean up resources gracefully."""
        # Currently a placeholder for future resource management.
        # If models or external handles need closing, add here.
        try:
            # Example: if cross encoder has a cleanup method
            pass
        except Exception as e:
            logger.error(f"Error during close(): {e}")
