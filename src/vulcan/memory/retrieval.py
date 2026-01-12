"""Memory retrieval, search, and indexing"""

from .base import Memory, MemoryQuery
from ..security_fixes import safe_pickle_load
import logging
import math
import pickle
import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# FAISS for vector search - Enhanced with CPU capability detection
try:
    from src.utils.faiss_config import initialize_faiss

    faiss, HAS_FAISS, _ = initialize_faiss()
except ImportError:
    # Fallback if faiss_config module is not available
    try:
        import faiss

        logging.info("FAISS library imported successfully")
        HAS_FAISS = True
    except (ImportError, ModuleNotFoundError) as e:
        logging.warning(
            f"Could not import FAISS: {e}. Falling back to NumPy-based retrieval"
        )
        HAS_FAISS = False
        faiss = None  # Define faiss as None so references don't crash

# Maintain backward compatibility with existing FAISS_AVAILABLE usage
FAISS_AVAILABLE = HAS_FAISS

# Additional indexing libraries
try:
    from whoosh.fields import ID, NUMERIC, TEXT, Schema
    from whoosh.index import create_in, open_dir
    from whoosh.qparser import QueryParser
    from whoosh.writing import AsyncWriter

    WHOOSH_AVAILABLE = True
except ImportError:
    WHOOSH_AVAILABLE = False
    logging.warning("Whoosh not available, text search limited")

# PyTorch for learned attention
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


logger = logging.getLogger(__name__)

# ============================================================
# RESULT CLASSES
# ============================================================


@dataclass
class RetrievalResult:
    """Result from memory retrieval operation."""

    memory_id: str
    score: float
    memory: Optional[Memory] = None
    metadata: Optional[Dict[str, Any]] = None
    relevance: Optional[float] = None
    context: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "memory_id": self.memory_id,
            "score": self.score,
            "memory": (
                self.memory.to_dict()
                if self.memory and hasattr(self.memory, "to_dict")
                else str(self.memory)
            ),
            "metadata": self.metadata,
            "relevance": self.relevance,
            "context": self.context,
        }


# ============================================================
# NUMPY FALLBACK INDEX
# ============================================================


class NumpyIndex:
    """Numpy-based vector index as FAISS fallback."""

    def __init__(self, dimension: int = 512):
        self.dimension = dimension
        self.embeddings = []
        self.memory_ids = []
        self.lock = threading.RLock()

    def add(self, memory_id: str, embedding: np.ndarray) -> bool:
        """Add embedding to index."""
        with self.lock:
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)

            # Normalize
            embedding = embedding / (np.linalg.norm(embedding) + 1e-10)

            self.embeddings.append(embedding.squeeze())
            self.memory_ids.append(memory_id)
            return True

    def search(
        self, query_embedding: np.ndarray, k: int = 10
    ) -> List[Tuple[str, float]]:
        """Search using cosine similarity."""
        with self.lock:
            if not self.embeddings:
                return []

            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)

            # Normalize
            query_embedding = query_embedding / (
                np.linalg.norm(query_embedding) + 1e-10
            )
            query_embedding = query_embedding.squeeze()

            # Compute similarities
            embeddings_matrix = np.array(self.embeddings)
            similarities = np.dot(embeddings_matrix, query_embedding)

            # Get top k
            top_indices = np.argsort(similarities)[-k:][::-1]

            results = []
            for idx in top_indices:
                if idx < len(self.memory_ids):
                    results.append((self.memory_ids[idx], float(similarities[idx])))

            return results

    def remove(self, memory_id: str) -> bool:
        """Remove from index."""
        with self.lock:
            try:
                idx = self.memory_ids.index(memory_id)
                del self.embeddings[idx]
                del self.memory_ids[idx]
                return True
            except ValueError:
                return False

    def clear(self):
        """Clear the index."""
        with self.lock:
            self.embeddings.clear()
            self.memory_ids.clear()

    def save(self, path: str):
        """Save NumPy index to disk."""
        with self.lock:
            data = {
                "embeddings": self.embeddings,
                "memory_ids": self.memory_ids,
                "dimension": self.dimension,
            }
            with open(f"{path}.numpy", "wb") as f:
                pickle.dump(data, f)

    def load(self, path: str):
        """Load NumPy index from disk."""
        with self.lock:
            numpy_file = f"{path}.numpy"
            if Path(numpy_file).exists():
                with open(numpy_file, "rb") as f:
                    data = safe_pickle_load(f)
                    self.embeddings = data["embeddings"]
                    self.memory_ids = data["memory_ids"]
                    self.dimension = data.get("dimension", 512)


# ============================================================
# ENHANCED MEMORY INDEX
# ============================================================


class MemoryIndex:
    """Vector index for fast similarity search with multiple backend support."""

    def __init__(
        self, dimension: int = 512, index_type: str = "flat", use_gpu: bool = False
    ):
        self.dimension = dimension
        self.index_type = index_type
        self.use_gpu = use_gpu and FAISS_AVAILABLE

        # FIXED: Check both FAISS_AVAILABLE and faiss is not None before using FAISS
        if FAISS_AVAILABLE and faiss is not None:
            try:
                self.index = self._create_faiss_index()
                self.is_faiss = True
                logger.info(
                    f"Created FAISS index: type={index_type}, dimension={dimension}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to create FAISS index: {e}, falling back to NumPy"
                )
                self.index = NumpyIndex(dimension)
                self.is_faiss = False
        else:
            self.index = NumpyIndex(dimension)
            self.is_faiss = False
            logger.info(
                f"Using NumPy index (FAISS not available): dimension={dimension}"
            )

        # Mapping from index position to memory ID
        self.id_map: List[str] = []
        self.reverse_map: Dict[str, int] = {}

        # Track deleted indices for periodic cleanup
        self.deleted_indices: Set[int] = set()
        self.rebuild_threshold = 100  # Rebuild after this many deletions

        self.lock = threading.RLock()

    def _create_faiss_index(self):
        """Create FAISS index based on type."""
        # FIXED: Check both FAISS_AVAILABLE flag AND that faiss module is not None
        if not FAISS_AVAILABLE or faiss is None:
            raise ImportError("FAISS is not available")

        if self.index_type == "flat":
            index = faiss.IndexFlatIP(self.dimension)  # Inner product
        elif self.index_type == "hnsw":
            index = faiss.IndexHNSWFlat(self.dimension, 32)
            index.hnsw.efConstruction = 40
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
            index.nprobe = 10
        elif self.index_type == "lsh":
            index = faiss.IndexLSH(self.dimension, self.dimension * 2)
        elif self.index_type == "pq":
            # Product quantization for compression
            index = faiss.IndexPQ(self.dimension, 16, 8)
        else:
            index = faiss.IndexFlatIP(self.dimension)

        # Move to GPU if requested and available
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                logger.info("Using GPU for FAISS index")
            except Exception as e:
                logger.warning(f"GPU not available for FAISS: {e}")

        return index

    def add(self, memory_id: str, embedding: np.ndarray) -> bool:
        """Add embedding to index."""
        with self.lock:
            try:
                if not self.is_faiss:
                    return self.index.add(memory_id, embedding)

                # FAISS implementation
                if embedding.ndim == 1:
                    embedding = embedding.reshape(1, -1)

                # Normalize for cosine similarity
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm

                # Check if memory already exists
                if memory_id in self.reverse_map:
                    # Update existing
                    idx = self.reverse_map[memory_id]
                    # FAISS doesn't support update, so mark for rebuild
                    self.deleted_indices.add(idx)

                # Add to index
                self.index.add(embedding.astype(np.float32))

                # Update mappings
                idx = len(self.id_map)
                self.id_map.append(memory_id)
                self.reverse_map[memory_id] = idx

                # Check if rebuild needed
                if len(self.deleted_indices) >= self.rebuild_threshold:
                    self._rebuild_index()

                return True

            except Exception as e:
                logger.error(f"Failed to add to index: {e}")
                return False

    def search(
        self, query_embedding: np.ndarray, k: int = 10
    ) -> List[Tuple[str, float]]:
        """Search for similar embeddings."""
        with self.lock:
            try:
                if not self.is_faiss:
                    return self.index.search(query_embedding, k)

                # FAISS implementation
                if len(self.id_map) == 0:
                    return []

                # Ensure correct shape
                if query_embedding.ndim == 1:
                    query_embedding = query_embedding.reshape(1, -1)

                # Normalize
                norm = np.linalg.norm(query_embedding)
                if norm > 0:
                    query_embedding = query_embedding / norm

                # Search for more than k to account for deleted items
                search_k = min(k + len(self.deleted_indices), len(self.id_map))

                distances, indices = self.index.search(
                    query_embedding.astype(np.float32), search_k
                )

                # Filter out deleted indices and convert to memory IDs
                results = []
                for i, idx in enumerate(indices[0]):
                    if (
                        0 <= idx < len(self.id_map)
                        and idx not in self.deleted_indices
                        and self.id_map[idx] is not None
                    ):
                        memory_id = self.id_map[idx]
                        score = float(distances[0][i])
                        results.append((memory_id, score))

                        if len(results) >= k:
                            break

                return results

            except Exception as e:
                logger.error(f"Failed to search index: {e}")
                return []

    def remove(self, memory_id: str) -> bool:
        """Remove from index."""
        with self.lock:
            if not self.is_faiss:
                return self.index.remove(memory_id)

            if memory_id in self.reverse_map:
                idx = self.reverse_map[memory_id]

                # Mark as deleted
                self.deleted_indices.add(idx)
                self.id_map[idx] = None
                del self.reverse_map[memory_id]

                # Rebuild if too many deletions
                if len(self.deleted_indices) >= self.rebuild_threshold:
                    self._rebuild_index()

                return True

            return False

    def clear(self):
        """Clear all memories from index."""
        with self.lock:
            if not self.is_faiss:
                if hasattr(self.index, 'clear'):
                    self.index.clear()
                return
            
            # For FAISS, recreate the index
            self.index = self._create_faiss_index()
            self.id_map.clear()
            self.reverse_map.clear()
            self.deleted_indices.clear()
            logger.info("Cleared memory index")

    def _rebuild_index(self):
        """Rebuild index without deleted items."""
        if not self.is_faiss:
            return

        logger.info(f"Rebuilding index (removed {len(self.deleted_indices)} items)")

        # Collect valid embeddings

        for i, memory_id in enumerate(self.id_map):
            if i not in self.deleted_indices and memory_id is not None:
                # Reconstruct embedding (would need to store them)
                # For now, skip rebuild if we can't reconstruct
                pass

        # Clear deleted indices
        self.deleted_indices.clear()

    def rebuild(self, memories: List[Tuple[str, np.ndarray]]):
        """Rebuild index from scratch."""
        with self.lock:
            # FIXED: Check both FAISS_AVAILABLE and faiss before using FAISS methods
            if self.is_faiss and FAISS_AVAILABLE and faiss is not None:
                # Clear existing
                self.index.reset()
                self.id_map = []
                self.reverse_map = {}
                self.deleted_indices.clear()
            else:
                self.index.clear()

            # Add all memories
            for memory_id, embedding in memories:
                self.add(memory_id, embedding)

    def save(self, path: str):
        """Save index to disk."""
        with self.lock:
            # FIXED: Check both FAISS_AVAILABLE and faiss before saving FAISS index
            if self.is_faiss and FAISS_AVAILABLE and faiss is not None:
                try:
                    faiss.write_index(self.index, f"{path}.index")
                    logger.info(f"FAISS index saved to {path}.index")
                except Exception as e:
                    logger.error(f"Failed to save FAISS index: {e}")
            else:
                # Save NumPy index
                try:
                    self.index.save(path)
                    logger.info(f"NumPy index saved to {path}.numpy")
                except Exception as e:
                    logger.error(f"Failed to save NumPy index: {e}")

            # Save mappings
            with open(f"{path}.map", "wb") as f:
                pickle.dump(
                    {
                        "id_map": self.id_map,
                        "reverse_map": self.reverse_map,
                        "deleted_indices": self.deleted_indices,
                        "is_faiss": self.is_faiss,
                        "dimension": self.dimension,
                    },
                    f,
                )

    def load(self, path: str):
        """Load index from disk."""
        with self.lock:
            # Load mappings
            map_file = f"{path}.map"
            if Path(map_file).exists():
                with open(map_file, "rb") as f:
                    data = safe_pickle_load(f)
                    self.id_map = data["id_map"]
                    self.reverse_map = data["reverse_map"]
                    self.deleted_indices = data.get("deleted_indices", set())
                    self.dimension = data.get("dimension", 512)
                    data.get("is_faiss", False)

            # FIXED: Check both FAISS_AVAILABLE and faiss before loading FAISS index
            if (
                self.is_faiss
                and FAISS_AVAILABLE
                and faiss is not None
                and Path(f"{path}.index").exists()
            ):
                try:
                    self.index = faiss.read_index(f"{path}.index")
                    logger.info(f"FAISS index loaded from {path}.index")
                except Exception as e:
                    logger.error(f"Failed to load FAISS index: {e}")
                    # Fallback to NumPy
                    self.index = NumpyIndex(self.dimension)
                    self.is_faiss = False
            else:
                # Load NumPy index
                try:
                    if not self.is_faiss and hasattr(self.index, "load"):
                        self.index.load(path)
                        logger.info(f"NumPy index loaded from {path}.numpy")
                except Exception as e:
                    logger.error(f"Failed to load NumPy index: {e}")


# ============================================================
# SHARDED MEMORY INDEX
# ============================================================


class ShardedMemoryIndex:
    """Sharded memory index for scalable vector search."""
    
    def __init__(
        self,
        dimension: int = 512,
        shard_size: int = 100000,
        index_type: str = "flat",
        use_gpu: bool = False,
    ):
        """Initialize sharded index.
        
        Args:
            dimension: Embedding dimension
            shard_size: Maximum items per shard
            index_type: Type of index to use per shard
            use_gpu: Whether to use GPU
        """
        self.dimension = dimension
        self.shard_size = shard_size
        self.index_type = index_type
        self.use_gpu = use_gpu
        
        # Shards: List of MemoryIndex instances
        self.shards: List[MemoryIndex] = []
        # Track which shard each memory is in
        self.memory_to_shard: Dict[str, int] = {}
        # Current shard for new additions
        self.current_shard_idx = 0
        
        self._lock = threading.RLock()
        
        # Create initial shard
        self._create_new_shard()
        
        logger.info(
            f"Initialized sharded index: dimension={dimension}, "
            f"shard_size={shard_size}, shards=1"
        )
    
    def _get_shard_size(self, shard: MemoryIndex) -> int:
        """Get the current size of a shard.
        
        Args:
            shard: MemoryIndex shard
            
        Returns:
            Number of memories in the shard
        """
        # For FAISS backend
        if hasattr(shard, 'id_map') and len(shard.id_map) > 0:
            return len(shard.id_map)
        # For NumPy backend
        elif hasattr(shard, 'index') and hasattr(shard.index, 'memory_ids'):
            return len(shard.index.memory_ids)
        # Fallback
        return 0
    
    def _create_new_shard(self) -> int:
        """Create a new shard.
        
        Returns:
            Index of the new shard
        """
        shard = MemoryIndex(
            dimension=self.dimension,
            index_type=self.index_type,
            use_gpu=self.use_gpu,
        )
        self.shards.append(shard)
        return len(self.shards) - 1
    
    def add(self, memory_id: str, embedding: np.ndarray) -> bool:
        """Add embedding to sharded index.
        
        Args:
            memory_id: Memory identifier
            embedding: Embedding vector
            
        Returns:
            True if added successfully
        """
        with self._lock:
            # Check if current shard is full
            current_shard = self.shards[self.current_shard_idx]
            current_size = self._get_shard_size(current_shard)
            
            if current_size >= self.shard_size:
                # Create new shard
                self.current_shard_idx = self._create_new_shard()
                current_shard = self.shards[self.current_shard_idx]
                logger.info(
                    f"Created new shard {self.current_shard_idx}, "
                    f"total shards: {len(self.shards)}"
                )
            
            # Add to current shard
            success = current_shard.add(memory_id, embedding)
            if success:
                self.memory_to_shard[memory_id] = self.current_shard_idx
            
            return success
    
    def search(
        self, query_embedding: np.ndarray, k: int = 10
    ) -> List[Tuple[str, float]]:
        """Search across all shards in parallel.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of (memory_id, score) tuples
        """
        with self._lock:
            if not self.shards:
                return []
            
            # Search all shards in parallel
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            all_results = []
            
            with ThreadPoolExecutor(max_workers=min(len(self.shards), 10)) as executor:
                # Submit search tasks for each shard
                future_to_shard = {
                    executor.submit(shard.search, query_embedding, k): idx
                    for idx, shard in enumerate(self.shards)
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_shard):
                    try:
                        shard_results = future.result()
                        all_results.extend(shard_results)
                    except Exception as e:
                        shard_idx = future_to_shard[future]
                        logger.error(f"Shard {shard_idx} search failed: {e}")
            
            # Sort by score and return top k
            all_results.sort(key=lambda x: x[1], reverse=True)
            return all_results[:k]
    
    def remove(self, memory_id: str) -> bool:
        """Remove memory from sharded index.
        
        Args:
            memory_id: Memory identifier
            
        Returns:
            True if removed successfully
        """
        with self._lock:
            if memory_id not in self.memory_to_shard:
                return False
            
            shard_idx = self.memory_to_shard[memory_id]
            if shard_idx >= len(self.shards):
                return False
            
            success = self.shards[shard_idx].remove(memory_id)
            if success:
                del self.memory_to_shard[memory_id]
            
            return success
    
    def clear(self):
        """Clear all shards."""
        with self._lock:
            for shard in self.shards:
                shard.clear()
            self.memory_to_shard.clear()
            logger.info("Cleared all shards")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get shard statistics.
        
        Returns:
            Dict with shard statistics
        """
        with self._lock:
            shard_utilization = [self._get_shard_size(shard) for shard in self.shards]
            return {
                'total_shards': len(self.shards),
                'shard_size': self.shard_size,
                'total_memories': len(self.memory_to_shard),
                'shard_utilization': shard_utilization,
                'avg_utilization': (
                    sum(shard_utilization) / len(self.shards)
                    if self.shards else 0
                ),
            }


# ============================================================
# TEXT SEARCH INDEX
# ============================================================


class TextSearchIndex:
    """Full-text search index for memory content."""

    def __init__(self, index_dir: str = "./text_index"):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        if WHOOSH_AVAILABLE:
            self._init_whoosh()
        else:
            self._init_simple_index()

    def _init_whoosh(self):
        """Initialize Whoosh full-text search."""
        self.schema = Schema(
            memory_id=ID(stored=True, unique=True),
            content=TEXT(stored=False),
            type=ID(stored=True),
            timestamp=NUMERIC(stored=True),
            importance=NUMERIC(stored=True),
        )

        if not self.index_dir.exists() or not any(self.index_dir.iterdir()):
            self.index = create_in(str(self.index_dir), self.schema)
        else:
            self.index = open_dir(str(self.index_dir))

        # Memory cache for Whoosh (since Whoosh doesn't store Memory objects)
        self.memory_cache = {}
        self.lock = threading.Lock()

    def _init_simple_index(self):
        """Initialize simple inverted index."""
        self.inverted_index = defaultdict(set)
        self.documents = {}
        self.lock = threading.Lock()

    def add(self, memory: Memory):
        """Add memory to text index."""
        if WHOOSH_AVAILABLE:
            self._add_whoosh(memory)
        else:
            self._add_simple(memory)

    def _add_whoosh(self, memory: Memory):
        """Add using Whoosh."""
        writer = AsyncWriter(self.index)
        writer.update_document(
            memory_id=memory.id,
            content=str(memory.content),
            type=memory.type.value,
            timestamp=memory.timestamp,
            importance=memory.importance,
        )
        writer.commit()

        # Store memory in cache for retrieval
        with self.lock:
            self.memory_cache[memory.id] = memory

    def _add_simple(self, memory: Memory):
        """Add using simple inverted index."""
        with self.lock:
            # Tokenize content
            content = str(memory.content).lower()
            tokens = re.findall(r"\w+", content)

            # Update inverted index
            for token in tokens:
                self.inverted_index[token].add(memory.id)

            # Store document with both list and set of tokens
            self.documents[memory.id] = {
                "content": content,
                "tokens": tokens,  # Store as list for counting
                "tokens_set": set(tokens),  # Keep set for fast lookup
                "memory": memory,
            }

    def search(self, query: str, limit: int = 10) -> List[Tuple[Memory, float]]:
        """Search for memories containing query text."""
        if WHOOSH_AVAILABLE:
            return self._search_whoosh(query, limit)
        else:
            return self._search_simple(query, limit)

    def _search_whoosh(self, query: str, limit: int) -> List[Tuple[Memory, float]]:
        """Search using Whoosh."""
        results = []

        with self.index.searcher() as searcher:
            query_obj = QueryParser("content", self.index.schema).parse(query)
            search_results = searcher.search(query_obj, limit=limit)

            for hit in search_results:
                memory_id = hit["memory_id"]
                # Retrieve memory object from cache
                with self.lock:
                    if memory_id in self.memory_cache:
                        results.append((self.memory_cache[memory_id], hit.score))

        return results

    def _search_simple(self, query: str, limit: int) -> List[Tuple[Memory, float]]:
        """Search using simple inverted index."""
        with self.lock:
            # Tokenize query
            query_tokens = set(re.findall(r"\w+", query.lower()))

            # Find matching documents
            matches = []
            for token in query_tokens:
                if token in self.inverted_index:
                    for doc_id in self.inverted_index[token]:
                        if doc_id in self.documents:
                            doc = self.documents[doc_id]
                            # Use token list for counting
                            # Calculate simple TF-IDF score
                            token_count = doc["tokens"].count(
                                token
                            )  # Count occurrences in list
                            tf = token_count / max(
                                1, len(doc["tokens"])
                            )  # Term frequency
                            idf = math.log(
                                len(self.documents)
                                / max(1, len(self.inverted_index[token]))
                            )
                            score = tf * idf
                            matches.append((doc["memory"], score))

            # Aggregate scores by document
            doc_scores = defaultdict(float)
            doc_memories = {}
            for memory, score in matches:
                doc_scores[memory.id] += score
                doc_memories[memory.id] = memory

            # Sort by score
            sorted_results = sorted(
                doc_scores.items(), key=lambda x: x[1], reverse=True
            )

            results = []
            for doc_id, score in sorted_results[:limit]:
                results.append((doc_memories[doc_id], score))

            return results

    def remove(self, memory_id: str):
        """Remove memory from index."""
        if WHOOSH_AVAILABLE:
            writer = AsyncWriter(self.index)
            writer.delete_by_term("memory_id", memory_id)
            writer.commit()
            # Remove from memory cache
            with self.lock:
                self.memory_cache.pop(memory_id, None)
        else:
            with self.lock:
                if memory_id in self.documents:
                    doc = self.documents[memory_id]
                    # Remove from inverted index using tokens_set
                    for token in doc["tokens_set"]:
                        self.inverted_index[token].discard(memory_id)
                    # Remove document
                    del self.documents[memory_id]


# ============================================================
# TEMPORAL INDEX
# ============================================================


class TemporalIndex:
    """Temporal indexing for efficient time-based queries."""

    def __init__(self):
        self.time_index = []  # Sorted list of (timestamp, memory_id)
        self.memory_map = {}  # memory_id -> memory
        self.lock = threading.RLock()

        # Time buckets for faster range queries
        self.hourly_buckets = defaultdict(list)
        self.daily_buckets = defaultdict(list)
        self.monthly_buckets = defaultdict(list)

    def add(self, memory: Memory):
        """Add memory to temporal index."""
        with self.lock:
            # Add to sorted list
            import bisect

            bisect.insort(self.time_index, (memory.timestamp, memory.id))

            # Add to map
            self.memory_map[memory.id] = memory

            # Add to time buckets
            from datetime import datetime

            dt = datetime.fromtimestamp(memory.timestamp)

            hour_key = dt.strftime("%Y-%m-%d-%H")
            day_key = dt.strftime("%Y-%m-%d")
            month_key = dt.strftime("%Y-%m")

            self.hourly_buckets[hour_key].append(memory.id)
            self.daily_buckets[day_key].append(memory.id)
            self.monthly_buckets[month_key].append(memory.id)

    def search_range(
        self, start_time: float, end_time: float, limit: Optional[int] = None
    ) -> List[Memory]:
        """Search memories within time range."""
        with self.lock:
            # FIX #MEM-9: Binary search on timestamps only to avoid type comparison issues
            # Previous code used (end_time, "zzz") assuming "zzz" is larger than all memory IDs,
            # which fails for IDs starting with 'z', '{', '~', etc.
            # 
            # New approach: Extract timestamps for binary search, then filter by range
            import bisect
            
            if not self.time_index:
                return []
            
            # Find start position: first timestamp >= start_time
            start_idx = bisect.bisect_left(self.time_index, (start_time, ""))
            
            # Get memories in range by iterating and checking timestamp
            results = []
            for i in range(start_idx, len(self.time_index)):
                timestamp, memory_id = self.time_index[i]
                
                # Stop if we've passed the end time
                if timestamp > end_time:
                    break
                    
                if memory_id in self.memory_map:
                    results.append(self.memory_map[memory_id])

                    if limit and len(results) >= limit:
                        break

            return results

    def search_recent(
        self, hours: float = 24, limit: Optional[int] = None
    ) -> List[Memory]:
        """Search recent memories."""
        end_time = time.time()
        start_time = end_time - (hours * 3600)
        return self.search_range(start_time, end_time, limit)

    def search_by_bucket(self, bucket_type: str, bucket_key: str) -> List[Memory]:
        """Search by time bucket (hour, day, month)."""
        with self.lock:
            if bucket_type == "hour":
                memory_ids = self.hourly_buckets.get(bucket_key, [])
            elif bucket_type == "day":
                memory_ids = self.daily_buckets.get(bucket_key, [])
            elif bucket_type == "month":
                memory_ids = self.monthly_buckets.get(bucket_key, [])
            else:
                return []

            results = []
            for memory_id in memory_ids:
                if memory_id in self.memory_map:
                    results.append(self.memory_map[memory_id])

            return results

    def remove(self, memory_id: str):
        """Remove memory from temporal index."""
        with self.lock:
            if memory_id not in self.memory_map:
                return

            memory = self.memory_map[memory_id]

            # Remove from sorted list
            self.time_index = [
                (t, mid) for t, mid in self.time_index if mid != memory_id
            ]

            # Remove from map
            del self.memory_map[memory_id]

            # Remove from buckets
            from datetime import datetime

            dt = datetime.fromtimestamp(memory.timestamp)

            hour_key = dt.strftime("%Y-%m-%d-%H")
            day_key = dt.strftime("%Y-%m-%d")
            month_key = dt.strftime("%Y-%m")

            if hour_key in self.hourly_buckets:
                self.hourly_buckets[hour_key] = [
                    mid for mid in self.hourly_buckets[hour_key] if mid != memory_id
                ]
            if day_key in self.daily_buckets:
                self.daily_buckets[day_key] = [
                    mid for mid in self.daily_buckets[day_key] if mid != memory_id
                ]
            if month_key in self.monthly_buckets:
                self.monthly_buckets[month_key] = [
                    mid for mid in self.monthly_buckets[month_key] if mid != memory_id
                ]


# ============================================================
# LEARNED ATTENTION MECHANISM
# ============================================================

if TORCH_AVAILABLE:

    class LearnedAttention(nn.Module):
        """Learned attention mechanism using neural network."""

        def __init__(
            self, input_dim: int = 512, hidden_dim: int = 256, num_heads: int = 8
        ):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.num_heads = num_heads
            self.head_dim = hidden_dim // num_heads

            # Multi-head attention layers
            self.W_q = nn.Linear(input_dim, hidden_dim)
            self.W_k = nn.Linear(input_dim, hidden_dim)
            self.W_v = nn.Linear(input_dim, hidden_dim)

            # Output projection
            self.W_o = nn.Linear(hidden_dim, input_dim)

            # Layer normalization
            self.layer_norm = nn.LayerNorm(input_dim)

            # Feed-forward network
            self.ffn = nn.Sequential(
                nn.Linear(input_dim, hidden_dim * 4),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim * 4, input_dim),
            )

        def forward(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
            """Compute multi-head attention."""
            batch_size = query.size(0)
            seq_len = keys.size(1)

            # Compute Q, K, V
            Q = self.W_q(query).view(batch_size, 1, self.num_heads, self.head_dim)
            K = self.W_k(keys).view(batch_size, seq_len, self.num_heads, self.head_dim)
            V = self.W_v(keys).view(batch_size, seq_len, self.num_heads, self.head_dim)

            # Transpose for attention computation
            Q = Q.transpose(1, 2)  # [batch, heads, 1, head_dim]
            K = K.transpose(1, 2)  # [batch, heads, seq_len, head_dim]
            V = V.transpose(1, 2)  # [batch, heads, seq_len, head_dim]

            # Compute attention scores
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attention_weights = torch.softmax(scores, dim=-1)

            # Apply attention to values
            context = torch.matmul(attention_weights, V)

            # Reshape and project
            context = (
                context.transpose(1, 2)
                .contiguous()
                .view(batch_size, -1, self.hidden_dim)
            )
            output = self.W_o(context)

            # Add residual and normalize
            output = self.layer_norm(output + query.unsqueeze(1))

            # Feed-forward network
            output = output + self.ffn(output)

            return output.squeeze(1), attention_weights.mean(dim=1).squeeze(1)

else:
    # Stub class when torch is not available
    LearnedAttention = None


class AttentionMechanism:
    """Enhanced attention mechanism for memory retrieval."""

    def __init__(self, hidden_dim: int = 256, input_dim: int = 512):
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        if TORCH_AVAILABLE:
            self.learned_attention = LearnedAttention(input_dim, hidden_dim)
            self.learned_attention.eval()  # Set to evaluation mode

            # Load pre-trained weights if available
            self._load_pretrained_weights()
        else:
            # Fallback to simple parameterized attention
            np.random.seed(42)  # For reproducibility
            self.W_q = np.random.randn(input_dim, hidden_dim) * 0.01
            self.W_k = np.random.randn(input_dim, hidden_dim) * 0.01
            self.W_v = np.random.randn(input_dim, hidden_dim) * 0.01

            # Learned temperature parameter
            self.temperature = 1.0

    def _load_pretrained_weights(self):
        """Load pre-trained attention weights if available."""
        weights_path = Path("./models/attention_weights.pt")
        if weights_path.exists():
            try:
                state_dict = torch.load(
                    weights_path, map_location="cpu", weights_only=True
                )
                self.learned_attention.load_state_dict(state_dict)
                logger.info("Loaded pre-trained attention weights")
            except Exception as e:
                logger.warning(f"Failed to load pre-trained attention weights: {e}")

    def compute_attention(
        self,
        query: np.ndarray,
        memories: List[np.ndarray],
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute attention weights for memories."""
        if not memories:
            return np.array([])

        # Ensure proper shapes
        if isinstance(memories[0], np.ndarray):
            memory_matrix = np.stack(memories)
        else:
            memory_matrix = np.array(memories)

        if query.ndim == 1:
            query = query.reshape(1, -1)

        # FIX: Validate dimension compatibility
        query_dim = query.shape[-1]
        memory_dim = memory_matrix.shape[-1]
        if query_dim != memory_dim:
            logger.warning(
                f"Dimension mismatch in attention: query={query_dim}, memories={memory_dim}"
            )
            # Return uniform weights instead of crashing
            return np.ones(len(memories)) / len(memories)

        if TORCH_AVAILABLE:
            return self._compute_learned_attention(query, memory_matrix, mask)
        else:
            return self._compute_simple_attention(query, memory_matrix, mask)

    def _compute_learned_attention(
        self, query: np.ndarray, keys: np.ndarray, mask: Optional[np.ndarray]
    ) -> np.ndarray:
        """Compute attention using learned neural network."""
        # Convert to tensors
        query_tensor = torch.tensor(query, dtype=torch.float32)
        keys_tensor = torch.tensor(keys, dtype=torch.float32).unsqueeze(0)

        # Forward pass
        with torch.no_grad():
            _, attention_weights = self.learned_attention(query_tensor, keys_tensor)

        # Convert back to numpy
        weights = attention_weights.numpy().squeeze()

        # Apply mask if provided
        if mask is not None:
            weights = weights * mask
            weights = weights / (weights.sum() + 1e-10)

        return weights

    def _compute_simple_attention(
        self, query: np.ndarray, keys: np.ndarray, mask: Optional[np.ndarray]
    ) -> np.ndarray:
        """Compute attention using parameterized matrices."""
        # FIX: Validate dimensions before matrix operations
        query_dim = query.shape[-1]
        if query_dim != self.input_dim:
            # FIX #MEM-5: Raise exception instead of returning uniform weights
            # Silent failure makes debugging impossible - users think search works
            # but get random results (uniform weights = all memories equally relevant)
            raise ValueError(
                f"Embedding dimension mismatch: query has {query_dim} dimensions "
                f"but attention mechanism expects {self.input_dim} dimensions. "
                f"Did you change the embedding model? "
                f"Regenerate all memory embeddings or use the same embedding model."
            )

        # Project query and keys
        Q = np.dot(query, self.W_q)
        K = np.dot(keys, self.W_k)  # FIX: Remove transpose - W_k is already (input_dim, hidden_dim)

        # Compute scaled dot-product attention
        scores = np.dot(K, Q.T).squeeze() / np.sqrt(self.hidden_dim)

        # Apply temperature scaling
        scores = scores / self.temperature

        # Apply mask if provided
        if mask is not None:
            scores = scores + (1 - mask) * -1e9

        # Softmax
        exp_scores = np.exp(scores - np.max(scores))
        attention_weights = exp_scores / (np.sum(exp_scores) + 1e-10)

        return attention_weights

    def apply_attention(
        self, memories: List[Memory], weights: np.ndarray, threshold: float = 0.01
    ) -> List[Memory]:
        """Apply attention weights to memories."""
        if len(memories) != len(weights):
            return memories

        # Filter out low-attention memories
        filtered_pairs = [(m, w) for m, w in zip(memories, weights) if w >= threshold]

        # Sort by attention weight
        filtered_pairs.sort(key=lambda x: x[1], reverse=True)

        return [m for m, _ in filtered_pairs]

    def train_on_feedback(
        self,
        queries: List[np.ndarray],
        memories: List[List[np.ndarray]],
        relevance_scores: List[List[float]],
    ):
        """Train attention mechanism on user feedback."""
        if not TORCH_AVAILABLE:
            logger.warning("Training requires PyTorch")
            return

        # Prepare training data
        optimizer = torch.optim.Adam(self.learned_attention.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        self.learned_attention.train()

        for query, mems, scores in zip(queries, memories, relevance_scores):
            query_tensor = torch.tensor(query, dtype=torch.float32).unsqueeze(0)
            keys_tensor = torch.tensor(np.array(mems), dtype=torch.float32).unsqueeze(0)
            target_scores = torch.tensor(scores, dtype=torch.float32)

            # Forward pass
            _, attention_weights = self.learned_attention(query_tensor, keys_tensor)

            # Compute loss
            loss = criterion(attention_weights.squeeze(), target_scores)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.learned_attention.eval()

        # Save trained weights
        self._save_weights()

    def _save_weights(self):
        """Save trained attention weights."""
        if TORCH_AVAILABLE:
            weights_path = Path("./models")
            weights_path.mkdir(exist_ok=True)
            torch.save(
                self.learned_attention.state_dict(),
                weights_path / "attention_weights.pt",
            )


# ============================================================
# MEMORY SEARCH
# ============================================================


class MemorySearch:
    """Advanced memory search with multiple index types."""

    def __init__(self, base_path: str = "./search_indices"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Vector indices for different embedding types
        self.indices: Dict[str, MemoryIndex] = {}

        # Text search
        self.text_index = TextSearchIndex(str(self.base_path / "text"))

        # Temporal index
        self.temporal_index = TemporalIndex()

        # Metadata index
        self.metadata_index = defaultdict(lambda: defaultdict(set))

        # Graph index for relationships
        self.graph_index = defaultdict(set)

        # Cache for search results
        self.cache = {}
        self.cache_size = 100

        self.lock = threading.RLock()

    def create_index(
        self, index_name: str, dimension: int = 512, index_type: str = "flat"
    ) -> MemoryIndex:
        """Create new search index."""
        index = MemoryIndex(dimension, index_type)
        self.indices[index_name] = index
        return index

    def add_memory(self, memory: Memory, index_name: str = "default"):
        """Add memory to all relevant indices."""
        with self.lock:
            # Add to vector index
            if memory.embedding is not None:
                if index_name not in self.indices:
                    dim = len(memory.embedding)
                    self.indices[index_name] = MemoryIndex(dim)

                self.indices[index_name].add(memory.id, memory.embedding)

            # Add to text index
            self.text_index.add(memory)

            # Add to temporal index
            self.temporal_index.add(memory)

            # Add to metadata index
            for key, value in memory.metadata.items():
                self.metadata_index[key][str(value)].add(memory.id)

            # Clear cache
            self.cache.clear()

    def semantic_search(
        self,
        query_embedding: np.ndarray,
        memories: Dict[str, Memory],
        k: int = 10,
        index_name: str = "default",
    ) -> List[Tuple[Memory, float]]:
        """Semantic similarity search."""
        # Check cache
        cache_key = f"semantic_{hash(query_embedding.tobytes())}_{k}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Get or create index
        if index_name not in self.indices:
            dim = len(query_embedding)
            self.indices[index_name] = MemoryIndex(dim)

        index = self.indices[index_name]

        # Add memories to index if needed
        for memory_id, memory in memories.items():
            if memory.embedding is not None and memory_id not in index.reverse_map:
                index.add(memory_id, memory.embedding)

        # Search
        results = index.search(query_embedding, k)

        # Convert to memories
        memory_results = []
        for memory_id, score in results:
            if memory_id in memories:
                memory_results.append((memories[memory_id], score))

        # Cache result
        self._update_cache(cache_key, memory_results)

        return memory_results

    def text_search(
        self, query: str, memories: Dict[str, Memory], limit: int = 10
    ) -> List[Tuple[Memory, float]]:
        """Full-text search."""
        # Ensure memories are in text index
        for memory in memories.values():
            # Check if already indexed (would need tracking)
            self.text_index.add(memory)

        return self.text_index.search(query, limit)

    def temporal_search(
        self,
        memories: Dict[str, Memory],
        time_range: Optional[Tuple[float, float]] = None,
        hours_back: Optional[float] = None,
        limit: int = 10,
    ) -> List[Memory]:
        """Search memories by time."""
        if hours_back is not None:
            return self.temporal_index.search_recent(hours_back, limit)
        elif time_range is not None:
            start_time, end_time = time_range
            return self.temporal_index.search_range(start_time, end_time, limit)
        else:
            # Return most recent
            return self.temporal_index.search_recent(24, limit)

    def metadata_search(
        self, memories: Dict[str, Memory], metadata_filters: Dict[str, Any]
    ) -> List[Memory]:
        """Search by metadata attributes."""
        matching_ids = None

        for key, value in metadata_filters.items():
            value_str = str(value)
            if key in self.metadata_index and value_str in self.metadata_index[key]:
                ids = self.metadata_index[key][value_str]

                if matching_ids is None:
                    matching_ids = ids.copy()
                else:
                    matching_ids &= ids

        if matching_ids is None:
            return []

        results = []
        for memory_id in matching_ids:
            if memory_id in memories:
                results.append(memories[memory_id])

        return results

    def causal_search(
        self,
        memories: Dict[str, Memory],
        cause_memory: Memory,
        max_time_delta: float = 3600,
        similarity_threshold: float = 0.7,
    ) -> List[Memory]:
        """Find causally related memories."""
        related = []

        for memory in memories.values():
            if memory.id == cause_memory.id:
                continue

            # Check temporal proximity (effect after cause)
            time_delta = memory.timestamp - cause_memory.timestamp
            if 0 < time_delta < max_time_delta:
                # Check semantic similarity
                if memory.embedding is not None and cause_memory.embedding is not None:
                    # Normalize embeddings
                    mem_norm = memory.embedding / (
                        np.linalg.norm(memory.embedding) + 1e-10
                    )
                    cause_norm = cause_memory.embedding / (
                        np.linalg.norm(cause_memory.embedding) + 1e-10
                    )

                    similarity = np.dot(mem_norm, cause_norm)
                    if similarity > similarity_threshold:
                        related.append(memory)
                else:
                    # Fallback to metadata matching
                    shared_keys = set(memory.metadata.keys()) & set(
                        cause_memory.metadata.keys()
                    )
                    if len(shared_keys) > len(cause_memory.metadata) * 0.5:
                        related.append(memory)

        # Sort by timestamp (causal order)
        related.sort(key=lambda m: m.timestamp)

        return related

    def pattern_search(
        self, memories: Dict[str, Memory], pattern: Dict[str, Any]
    ) -> List[Memory]:
        """Search for memories matching a complex pattern."""
        matching = []

        for memory in memories.values():
            if self._matches_pattern(memory, pattern):
                matching.append(memory)

        return matching

    def hybrid_search(
        self,
        query: MemoryQuery,
        memories: Dict[str, Memory],
        weights: Dict[str, float] = None,
    ) -> List[Tuple[Memory, float]]:
        """Hybrid search combining multiple search types."""
        if weights is None:
            weights = {"semantic": 0.4, "text": 0.2, "temporal": 0.2, "metadata": 0.2}

        all_scores = defaultdict(float)

        # Semantic search
        if query.embedding is not None and weights.get("semantic", 0) > 0:
            semantic_results = self.semantic_search(
                query.embedding, memories, query.limit * 2
            )
            for memory, score in semantic_results:
                all_scores[memory.id] += score * weights["semantic"]

        # Text search
        if query.content and weights.get("text", 0) > 0:
            text_results = self.text_search(
                str(query.content), memories, query.limit * 2
            )
            for memory, score in text_results:
                # Normalize text scores to [0, 1]
                normalized_score = min(1.0, score / 10.0)
                all_scores[memory.id] += normalized_score * weights["text"]

        # Temporal scoring
        if weights.get("temporal", 0) > 0:
            current_time = time.time()
            for memory in memories.values():
                # Recency score
                age = current_time - memory.timestamp
                recency_score = np.exp(-age / (7 * 24 * 3600))  # Decay over week
                all_scores[memory.id] += recency_score * weights["temporal"]

        # Metadata matching
        if query.filters and weights.get("metadata", 0) > 0:
            metadata_matches = self.metadata_search(memories, query.filters)
            for memory in metadata_matches:
                all_scores[memory.id] += weights["metadata"]

        # Sort by combined score
        sorted_results = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)

        # Convert to memory objects
        final_results = []
        for memory_id, score in sorted_results[: query.limit]:
            if memory_id in memories:
                final_results.append((memories[memory_id], score))

        return final_results

    def _matches_pattern(self, memory: Memory, pattern: Dict[str, Any]) -> bool:
        """Check if memory matches pattern."""
        for key, value in pattern.items():
            if key == "type" and memory.type != value:
                return False

            elif key == "min_importance" and memory.importance < value:
                return False

            elif key == "max_importance" and memory.importance > value:
                return False

            elif key == "metadata":
                for meta_key, meta_value in value.items():
                    if memory.metadata.get(meta_key) != meta_value:
                        return False

            elif key == "content_contains":
                if value not in str(memory.content):
                    return False

            elif key == "content_regex":
                import re

                if not re.search(value, str(memory.content)):
                    return False

        return True

    def _update_cache(self, key: str, value: Any):
        """Update search cache with LRU eviction."""
        self.cache[key] = value

        # Evict oldest if cache too large
        if len(self.cache) > self.cache_size:
            # Simple FIFO eviction (could use LRU)
            oldest = next(iter(self.cache))
            del self.cache[oldest]

    def save_indices(self):
        """Save all indices to disk."""
        for name, index in self.indices.items():
            index.save(str(self.base_path / name))

        # Save metadata index
        with open(self.base_path / "metadata_index.pkl", "wb") as f:
            pickle.dump(dict(self.metadata_index), f)

    def load_indices(self):
        """Load indices from disk."""
        # Load vector indices
        for index_file in self.base_path.glob("*.map"):
            name = index_file.stem
            index = MemoryIndex()
            index.load(str(self.base_path / name))
            self.indices[name] = index

        # Load metadata index
        metadata_file = self.base_path / "metadata_index.pkl"
        if metadata_file.exists():
            with open(metadata_file, "rb") as f:
                self.metadata_index = defaultdict(
                    lambda: defaultdict(set), safe_pickle_load(f)
                )
