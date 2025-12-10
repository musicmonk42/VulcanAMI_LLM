"""Memory persistence, compression, and versioning"""

import copy
import hashlib
import json
import logging
import os
import pickle
import shutil
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lz4.frame
import numpy as np

from .base import CompressionType, Memory, MemoryType

# Try to import advanced libraries
try:
    from cryptography.fernet import Fernet

    ENCRYPTION_AVAILABLE = True
except ImportError:
    ENCRYPTION_AVAILABLE = False
    logging.warning("Cryptography library not available, encryption disabled.")

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available, neural compression disabled")

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("Sentence transformers not available, semantic compression limited")

logger = logging.getLogger(__name__)

# ============================================================
# NEURAL COMPRESSION MODELS
# ============================================================

if TORCH_AVAILABLE:

    class NeuralCompressor(nn.Module):
        """Neural network for memory compression."""

        def __init__(self, input_dim: int = 1024, latent_dim: int = 128):
            super().__init__()
            self.input_dim = input_dim
            self.latent_dim = latent_dim

            # Encoder
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Linear(256, latent_dim),
                nn.Tanh(),
            )

            # Decoder
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Linear(512, input_dim),
                nn.Sigmoid(),
            )

        def encode(self, x: torch.Tensor) -> torch.Tensor:
            """Encode to latent space."""
            return self.encoder(x)

        def decode(self, z: torch.Tensor) -> torch.Tensor:
            """Decode from latent space."""
            return self.decoder(z)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """Forward pass returns reconstruction and latent."""
            z = self.encode(x)
            x_recon = self.decode(z)
            return x_recon, z
else:
    # Stub class when torch is not available
    NeuralCompressor = None


class SemanticCompressor:
    """Semantic compression using language models."""

    def __init__(self):
        self.model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as e:
                logger.debug(
                    f"{self.__class__.__name__ if hasattr(self, '__class__') else 'Operation'} error: {e}"
                )

        # Compression strategies
        self.strategies = {
            "summary": self._compress_summary,
            "keywords": self._compress_keywords,
            "embedding": self._compress_embedding,
        }

    def compress(self, content: Any, strategy: str = "embedding") -> Dict[str, Any]:
        """Compress content using semantic understanding."""
        if strategy not in self.strategies:
            strategy = "embedding"

        return self.strategies[strategy](content)

    def _compress_summary(self, content: Any) -> Dict[str, Any]:
        """Compress by generating summary."""
        text = str(content)

        # Simple extractive summarization
        sentences = text.split(".")
        if len(sentences) <= 3:
            summary = text
        else:
            # Take first, middle, and last sentences
            summary = ". ".join(
                [sentences[0], sentences[len(sentences) // 2], sentences[-1]]
            )

        return {
            "type": "summary",
            "compressed": summary,
            "original_length": len(text),
            "compressed_length": len(summary),
            "compression_ratio": len(text) / max(1, len(summary)),
        }

    def _compress_keywords(self, content: Any) -> Dict[str, Any]:
        """Compress by extracting keywords."""
        text = str(content)

        # Simple keyword extraction (TF-IDF would be better)
        words = text.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Skip short words
                word_freq[word] = word_freq.get(word, 0) + 1

        # Get top keywords
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "type": "keywords",
            "keywords": [k for k, v in keywords],
            "frequencies": dict(keywords),
            "original_length": len(text),
            "compressed_size": len(keywords) * 10,  # Approximate
        }

    def _compress_embedding(self, content: Any) -> Dict[str, Any]:
        """Compress to semantic embedding."""
        text = str(content)

        if self.model:
            # Generate embedding
            embedding = self.model.encode(text[:512])  # Truncate for model limits

            # Quantize to reduce size
            quantized = (embedding * 127).astype(np.int8)

            return {
                "type": "embedding",
                "embedding": quantized,
                "original_length": len(text),
                "embedding_dim": len(quantized),
                "compression_ratio": len(text) / len(quantized),
            }
        else:
            # Fallback: hash-based pseudo-embedding
            hash_val = hashlib.sha256(text.encode()).digest()
            embedding = np.frombuffer(hash_val, dtype=np.uint8).astype(np.int8)

            return {
                "type": "hash_embedding",
                "embedding": embedding,
                "original_length": len(text),
                "embedding_dim": len(embedding),
            }

    def decompress(
        self, compressed_data: Dict[str, Any], original_hint: Optional[str] = None
    ) -> str:
        """Decompress semantic data."""
        comp_type = compressed_data.get("type")

        if comp_type == "summary":
            # Summary is already readable
            return compressed_data["compressed"]

        elif comp_type == "keywords":
            # Reconstruct from keywords
            keywords = compressed_data["keywords"]
            return f"Content about: {', '.join(keywords)}"

        elif comp_type in ["embedding", "hash_embedding"]:
            # Can't fully reconstruct from embedding
            # Return a placeholder or use original hint
            if original_hint:
                return f"[Compressed content similar to: {original_hint[:100]}...]"
            return "[Compressed semantic content - embedding only]"

        return "[Unknown compression type]"


# ============================================================
# ENHANCED MEMORY COMPRESSION
# ============================================================


class MemoryCompressor:
    """Handles memory compression and decompression."""

    def __init__(self):
        self.neural_compressor = None
        self.semantic_compressor = None

        # Initialize neural compressor if available
        if TORCH_AVAILABLE:
            self.neural_compressor = NeuralCompressor()
            self.neural_compressor.eval()  # Set to evaluation mode

        # Initialize semantic compressor
        self.semantic_compressor = SemanticCompressor()

        # Cache for neural models
        self.compression_cache = {}
        self.cache_lock = threading.Lock()

    @staticmethod
    def compress(memory: Memory, compression_type: CompressionType) -> bytes:
        """Compress memory content."""
        # Serialize memory content for base compression
        serialized = pickle.dumps(memory.content)

        if compression_type == CompressionType.NONE:
            return serialized

        elif compression_type == CompressionType.LZ4:
            return lz4.frame.compress(serialized)

        elif compression_type == CompressionType.ZSTD:
            try:
                import zstandard as zstd

                cctx = zstd.ZstdCompressor(level=3)
                return cctx.compress(serialized)
            except ImportError:
                logger.warning("zstd not available, using lz4")
                return lz4.frame.compress(serialized)

        elif compression_type == CompressionType.NEURAL:
            return MemoryCompressor._neural_compress(memory)

        elif compression_type == CompressionType.SEMANTIC:
            return MemoryCompressor._semantic_compress(memory)

        else:
            return serialized

    @staticmethod
    def _neural_compress(memory: Memory) -> bytes:
        """Neural compression using autoencoder."""
        if not TORCH_AVAILABLE:
            # Fallback to LZ4
            return lz4.frame.compress(pickle.dumps(memory.content))

        try:
            # Get or create compressor
            compressor = MemoryCompressor._get_neural_compressor()

            # Prepare input
            if isinstance(memory.content, str):
                # Convert text to embedding
                text_bytes = memory.content.encode("utf-8")
                # Pad or truncate to fixed size
                fixed_size = 1024
                if len(text_bytes) < fixed_size:
                    text_bytes = text_bytes + b"\0" * (fixed_size - len(text_bytes))
                else:
                    text_bytes = text_bytes[:fixed_size]

                input_array = (
                    np.frombuffer(text_bytes, dtype=np.uint8).astype(np.float32) / 255.0
                )

            elif isinstance(memory.content, np.ndarray):
                # Use array directly
                input_array = memory.content.flatten()[:1024]
                if len(input_array) < 1024:
                    input_array = np.pad(input_array, (0, 1024 - len(input_array)))
                input_array = input_array.astype(np.float32)

            else:
                # Serialize and treat as bytes
                serialized = pickle.dumps(memory.content)
                fixed_size = 1024
                if len(serialized) < fixed_size:
                    serialized = serialized + b"\0" * (fixed_size - len(serialized))
                else:
                    serialized = serialized[:fixed_size]

                input_array = (
                    np.frombuffer(serialized, dtype=np.uint8).astype(np.float32) / 255.0
                )

            # Convert to tensor
            input_tensor = torch.tensor(input_array).unsqueeze(0)

            # Compress
            with torch.no_grad():
                _, latent = compressor(input_tensor)

            # Package compressed data
            compressed_data = {
                "latent": latent.numpy(),
                "original_type": type(memory.content).__name__,
                "original_shape": input_array.shape,
                "memory_metadata": {
                    "id": memory.id,
                    "type": memory.type.value,
                    "timestamp": memory.timestamp,
                },
            }

            return pickle.dumps(compressed_data)

        except Exception as e:
            logger.error(f"Neural compression failed: {e}")
            # Fallback to LZ4
            return lz4.frame.compress(pickle.dumps(memory.content))

    @staticmethod
    def _semantic_compress(memory: Memory) -> bytes:
        """Semantic compression for text content."""
        try:
            compressor = SemanticCompressor()

            # Compress based on content type
            if isinstance(memory.content, str):
                compressed = compressor.compress(memory.content, strategy="embedding")
            elif isinstance(memory.content, dict) and "text" in memory.content:
                compressed = compressor.compress(
                    memory.content["text"], strategy="embedding"
                )
            else:
                # For non-text, use summary strategy
                compressed = compressor.compress(
                    str(memory.content)[:1000], strategy="summary"
                )

            # Package with metadata
            result = {
                "compressed": compressed,
                "original_type": type(memory.content).__name__,
                "memory_metadata": {
                    "id": memory.id,
                    "type": memory.type.value,
                    "timestamp": memory.timestamp,
                    "importance": memory.importance,
                },
            }

            return pickle.dumps(result)

        except Exception as e:
            logger.error(f"Semantic compression failed: {e}")
            # Fallback to LZ4
            return lz4.frame.compress(pickle.dumps(memory.content))

    @staticmethod
    def decompress(data: bytes, compression_type: CompressionType) -> Any:
        """Decompress memory content."""
        if compression_type == CompressionType.NONE:
            return pickle.loads(data)

        elif compression_type == CompressionType.LZ4:
            decompressed = lz4.frame.decompress(data)
            return pickle.loads(decompressed)

        elif compression_type == CompressionType.ZSTD:
            try:
                import zstandard as zstd

                dctx = zstd.ZstdDecompressor()
                decompressed = dctx.decompress(data)
                return pickle.loads(decompressed)
            except ImportError:
                decompressed = lz4.frame.decompress(data)
                return pickle.loads(decompressed)

        elif compression_type == CompressionType.NEURAL:
            return MemoryCompressor._neural_decompress(data)

        elif compression_type == CompressionType.SEMANTIC:
            return MemoryCompressor._semantic_decompress(data)

        else:
            return pickle.loads(data)

    @staticmethod
    def _neural_decompress(data: bytes) -> Any:
        """Decompress neural compressed data."""
        if not TORCH_AVAILABLE:
            # Try to extract original from fallback
            try:
                decompressed = lz4.frame.decompress(data)
                return pickle.loads(decompressed)
            except Exception as e:
                return None

        try:
            compressed_data = pickle.loads(data)

            # Check if it's neural compressed
            if "latent" not in compressed_data:
                # Fallback format
                return compressed_data

            # Get compressor
            compressor = MemoryCompressor._get_neural_compressor()

            # Reconstruct from latent
            latent_tensor = torch.tensor(compressed_data["latent"])

            with torch.no_grad():
                reconstructed = compressor.decode(latent_tensor)

            # Convert back to original format
            reconstructed_array = reconstructed.numpy().squeeze()

            if compressed_data["original_type"] == "str":
                # Reconstruct string
                bytes_array = (reconstructed_array * 255).astype(np.uint8).tobytes()
                try:
                    result = bytes_array.decode("utf-8").rstrip("\0")
                except Exception as e:
                    result = f"[Neural reconstruction - metadata: {compressed_data['memory_metadata']}]"

            elif compressed_data["original_type"] == "ndarray":
                result = reconstructed_array

            else:
                # Try to unpickle
                bytes_array = (reconstructed_array * 255).astype(np.uint8).tobytes()
                try:
                    result = pickle.loads(bytes_array)
                except Exception as e:  # Return reconstruction with metadata
                    result = {
                        "reconstructed": reconstructed_array,
                        "metadata": compressed_data["memory_metadata"],
                    }

            return result

        except Exception as e:
            logger.error(f"Neural decompression failed: {e}")
            return None

    @staticmethod
    def _semantic_decompress(data: bytes) -> Any:
        """Decompress semantic compressed data."""
        try:
            result = pickle.loads(data)

            if "compressed" not in result:
                # Not semantic format
                return result

            compressed = result["compressed"]
            metadata = result["memory_metadata"]

            # Reconstruct based on compression type
            compressor = SemanticCompressor()

            if compressed["type"] == "summary":
                # Summary is directly usable
                content = compressed["compressed"]

            elif compressed["type"] == "keywords":
                # Reconstruct from keywords
                keywords = compressed["keywords"]
                content = f"[Keywords: {', '.join(keywords)}] - Memory {metadata['id']}"

            elif compressed["type"] in ["embedding", "hash_embedding"]:
                # Can't fully reconstruct, provide metadata
                content = (
                    f"[Semantic embedding - Memory {metadata['id']}, "
                    f"Type: {metadata['type']}, "
                    f"Importance: {metadata.get('importance', 'N/A')}]"
                )

            else:
                content = f"[Compressed content - {metadata}]"

            return content

        except Exception as e:
            logger.error(f"Semantic decompression failed: {e}")
            try:
                # Try LZ4 fallback
                decompressed = lz4.frame.decompress(data)
                return pickle.loads(decompressed)
            except Exception as e:
                return None

    @staticmethod
    def _get_neural_compressor():
        """Get or create neural compressor instance."""
        if not TORCH_AVAILABLE:
            return None

        # Simple singleton pattern
        if not hasattr(MemoryCompressor, "_neural_instance"):
            MemoryCompressor._neural_instance = NeuralCompressor()
            MemoryCompressor._neural_instance.eval()
        return MemoryCompressor._neural_instance

    @staticmethod
    def estimate_compression_ratio(
        memory: Memory, compression_type: CompressionType
    ) -> float:
        """Estimate compression ratio."""
        original_size = len(pickle.dumps(memory.content))
        compressed = MemoryCompressor.compress(memory, compression_type)
        compressed_size = len(compressed)

        if compressed_size > 0:
            return original_size / compressed_size
        return 1.0


# ============================================================
# MEMORY VERSION CONTROL
# ============================================================


@dataclass
class MemoryVersion:
    """Version of a memory."""

    version_id: str
    memory_id: str
    timestamp: float
    content_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_version: Optional[str] = None

    # Additional version info
    author: str = "system"
    message: str = ""
    tags: List[str] = field(default_factory=list)


class MemoryVersionControl:
    """Git-like version control for memories."""

    def __init__(self, storage_path: str = "./memory_versions"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Version database
        self.db_path = self.storage_path / "versions.db"
        self._init_database()

        # In-memory cache
        self.versions: Dict[str, List[MemoryVersion]] = {}
        self.current_versions: Dict[str, str] = {}
        self.branches: Dict[str, Dict[str, str]] = {"main": {}}
        self.current_branch = "main"

        self.lock = threading.RLock()

    def _init_database(self):
        """Initialize SQLite database for version tracking."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS versions (
                version_id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                content_hash TEXT NOT NULL,
                parent_version TEXT,
                author TEXT,
                message TEXT,
                branch TEXT,
                metadata TEXT
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memory_id ON versions(memory_id)
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS branches (
                name TEXT PRIMARY KEY,
                head_version TEXT,
                created_at REAL
            )
        """)

        conn.commit()
        conn.close()

    def create_version(
        self,
        memory: Memory,
        message: str = "",
        author: str = "system",
        parent_version: Optional[str] = None,
    ) -> str:
        """Create new version of memory with commit message."""
        with self.lock:
            # Generate version ID
            version_id = self._generate_version_id(memory)

            # Compute content hash
            content_hash = self._compute_content_hash(memory.content)

            # Check if content changed
            if memory.id in self.versions:
                last_version = self.versions[memory.id][-1]
                if last_version.content_hash == content_hash:
                    logger.debug(f"No changes in memory {memory.id}, skipping version")
                    return last_version.version_id

            # Create version record
            version = MemoryVersion(
                version_id=version_id,
                memory_id=memory.id,
                timestamp=time.time(),
                content_hash=content_hash,
                metadata=memory.metadata.copy(),
                parent_version=parent_version or self.current_versions.get(memory.id),
                author=author,
                message=message or f"Update memory {memory.id}",
            )

            # Store in memory
            if memory.id not in self.versions:
                self.versions[memory.id] = []
            self.versions[memory.id].append(version)
            self.current_versions[memory.id] = version_id

            # Store in database
            self._save_version_to_db(version)

            # Update branch head
            self.branches[self.current_branch][memory.id] = version_id

            return version_id

    def _save_version_to_db(self, version: MemoryVersion):
        """Save version to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO versions
            (version_id, memory_id, timestamp, content_hash, parent_version,
             author, message, branch, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                version.version_id,
                version.memory_id,
                version.timestamp,
                version.content_hash,
                version.parent_version,
                version.author,
                version.message,
                self.current_branch,
                json.dumps(version.metadata),
            ),
        )

        conn.commit()
        conn.close()

    def get_version(
        self, memory_id: str, version_id: Optional[str] = None
    ) -> Optional[MemoryVersion]:
        """Get specific version or current version."""
        with self.lock:
            if version_id is None:
                # Get current version for branch
                version_id = self.branches[self.current_branch].get(memory_id)
                if not version_id:
                    version_id = self.current_versions.get(memory_id)

            # Check cache
            if memory_id in self.versions:
                for version in self.versions[memory_id]:
                    if version.version_id == version_id:
                        return version

            # Check database
            return self._load_version_from_db(version_id)

    def _load_version_from_db(self, version_id: str) -> Optional[MemoryVersion]:
        """Load version from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM versions WHERE version_id = ?
        """,
            (version_id,),
        )

        row = cursor.fetchone()
        conn.close()

        if row:
            return MemoryVersion(
                version_id=row[0],
                memory_id=row[1],
                timestamp=row[2],
                content_hash=row[3],
                parent_version=row[4],
                author=row[5],
                message=row[6],
                metadata=json.loads(row[8]) if row[8] else {},
            )

        return None

    def get_history(self, memory_id: str, limit: int = 100) -> List[MemoryVersion]:
        """Get version history for memory."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM versions
            WHERE memory_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """,
            (memory_id, limit),
        )

        rows = cursor.fetchall()
        conn.close()

        history = []
        for row in rows:
            history.append(
                MemoryVersion(
                    version_id=row[0],
                    memory_id=row[1],
                    timestamp=row[2],
                    content_hash=row[3],
                    parent_version=row[4],
                    author=row[5],
                    message=row[6],
                    metadata=json.loads(row[8]) if row[8] else {},
                )
            )

        return history

    def create_branch(
        self, branch_name: str, from_branch: Optional[str] = None
    ) -> bool:
        """Create new branch."""
        with self.lock:
            if branch_name in self.branches:
                return False

            from_branch = from_branch or self.current_branch
            self.branches[branch_name] = self.branches[from_branch].copy()

            # Save to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO branches (name, created_at) VALUES (?, ?)
            """,
                (branch_name, time.time()),
            )
            conn.commit()
            conn.close()

            return True

    def switch_branch(self, branch_name: str) -> bool:
        """Switch to different branch."""
        with self.lock:
            if branch_name not in self.branches:
                return False

            self.current_branch = branch_name
            self.current_versions = self.branches[branch_name].copy()
            return True

    def merge_branches(
        self, from_branch: str, to_branch: Optional[str] = None
    ) -> Dict[str, Any]:
        """Merge branches."""
        with self.lock:
            to_branch = to_branch or self.current_branch

            if from_branch not in self.branches or to_branch not in self.branches:
                return {"success": False, "error": "Branch not found"}

            conflicts = []
            merged = 0

            for memory_id, version_id in self.branches[from_branch].items():
                if memory_id not in self.branches[to_branch]:
                    # Fast-forward
                    self.branches[to_branch][memory_id] = version_id
                    merged += 1
                elif self.branches[to_branch][memory_id] != version_id:
                    # Conflict
                    conflicts.append(memory_id)

            return {"success": True, "merged": merged, "conflicts": conflicts}

    def rollback(self, memory_id: str, version_id: str) -> bool:
        """Rollback to specific version."""
        with self.lock:
            version = self.get_version(memory_id, version_id)
            if version:
                self.current_versions[memory_id] = version_id
                self.branches[self.current_branch][memory_id] = version_id
                return True
            return False

    def diff_versions(self, version_id1: str, version_id2: str) -> Dict[str, Any]:
        """Compare two versions."""
        v1 = self._load_version_from_db(version_id1)
        v2 = self._load_version_from_db(version_id2)

        if not v1 or not v2:
            return {"error": "Version not found"}

        return {
            "version1": version_id1,
            "version2": version_id2,
            "hash_changed": v1.content_hash != v2.content_hash,
            "metadata_diff": self._diff_metadata(v1.metadata, v2.metadata),
            "time_diff": v2.timestamp - v1.timestamp,
        }

    def _diff_metadata(self, meta1: Dict, meta2: Dict) -> Dict[str, Any]:
        """Compare metadata dictionaries."""
        added = {k: v for k, v in meta2.items() if k not in meta1}
        removed = {k: v for k, v in meta1.items() if k not in meta2}
        changed = {
            k: (meta1[k], meta2[k])
            for k in meta1
            if k in meta2 and meta1[k] != meta2[k]
        }

        return {"added": added, "removed": removed, "changed": changed}

    def _generate_version_id(self, memory: Memory) -> str:
        """Generate unique version ID."""
        content = f"{memory.id}_{time.time()}_{id(memory)}_{self.current_branch}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _compute_content_hash(self, content: Any) -> str:
        """Compute hash of content."""
        serialized = pickle.dumps(content, protocol=pickle.HIGHEST_PROTOCOL)
        return hashlib.sha256(serialized).hexdigest()


# ============================================================
# MEMORY PERSISTENCE
# ============================================================


class MemoryPersistence:
    """Handles persistence of memories to disk with advanced features."""

    def __init__(self, base_path: str = "./memory_store"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Paths for different storage
        self.memories_path = self.base_path / "memories"
        self.index_path = self.base_path / "indices"
        self.metadata_path = self.base_path / "metadata"
        self.backups_path = self.base_path / "backups"

        for path in [
            self.memories_path,
            self.index_path,
            self.metadata_path,
            self.backups_path,
        ]:
            path.mkdir(exist_ok=True)

        self.compressor = MemoryCompressor()
        self.version_control = MemoryVersionControl(str(self.base_path / "versions"))

        # FIX 628: Stricter encryption key handling
        self.cipher = None
        self.encryption_key = None  # Store key for persistence across instances

        if ENCRYPTION_AVAILABLE:
            key = os.getenv("MEMORY_ENCRYPT_KEY")
            if key:
                try:
                    self.cipher = Fernet(key.encode())
                    self.encryption_key = key.encode()
                    logger.info("✓ Encryption enabled with provided key")
                except Exception as e:
                    logger.error(f"✗ Invalid encryption key: {e}")
                    raise ValueError("MEMORY_ENCRYPT_KEY is malformed or invalid")
            else:
                # FIX 628: Make ephemeral key mode explicit opt-in
                allow_ephemeral = (
                    os.getenv("ALLOW_EPHEMERAL_KEY", "false").lower() == "true"
                )

                if allow_ephemeral:
                    # Store ephemeral key to disk for this session
                    key_file = self.base_path / ".ephemeral_key"
                    if key_file.exists():
                        # Reuse existing ephemeral key
                        with open(key_file, "rb") as f:
                            self.encryption_key = f.read()
                        self.cipher = Fernet(self.encryption_key)
                        logger.info(
                            "Reusing ephemeral encryption key from this session"
                        )
                    else:
                        # Generate new ephemeral key
                        self.encryption_key = Fernet.generate_key()
                        self.cipher = Fernet(self.encryption_key)
                        # Save for reuse within same session
                        with open(key_file, "wb") as f:
                            f.write(self.encryption_key)
                        logger.warning("=" * 70)
                        logger.warning("⚠  USING EPHEMERAL ENCRYPTION KEY")
                        logger.warning(
                            "⚠  Memories will NOT be readable after process restart"
                        )
                        logger.warning(
                            "⚠  Set MEMORY_ENCRYPT_KEY environment variable for persistence"
                        )
                        logger.warning("=" * 70)
                else:
                    logger.info(
                        "Encryption disabled. Set MEMORY_ENCRYPT_KEY to enable."
                    )
                    logger.info(
                        "Or set ALLOW_EPHEMERAL_KEY=true for session-only encryption."
                    )
                    self.cipher = None
        else:
            logger.warning("cryptography library not available - encryption disabled")

        # Write buffer for batch operations
        self.write_buffer: List[Tuple[Memory, CompressionType]] = []
        self.buffer_lock = threading.Lock()
        self.buffer_size = 100
        self.flush_interval = 10.0  # seconds

        # FIX 629: Add shutdown event
        self._shutdown_event = threading.Event()

        # Background flushing
        self.flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self.flush_thread.start()

        # Checkpoint info
        self.last_checkpoint = time.time()
        self.checkpoint_interval = 300.0  # 5 minutes

        # Statistics
        self.stats = {
            "total_saves": 0,
            "total_loads": 0,
            "compression_ratio_avg": 1.0,
            "last_backup": None,
        }

    def _flush_loop(self):
        """Background thread to flush write buffer."""
        # FIX 629: Check shutdown event in loop
        while not self._shutdown_event.is_set():
            # Wait with timeout instead of sleep
            self._shutdown_event.wait(timeout=self.flush_interval)

            if not self._shutdown_event.is_set():
                try:
                    self.flush_buffer()
                except Exception as e:
                    logger.error(f"Flush error: {e}")

        # FIX 629: Final flush on shutdown
        logger.info("Performing final buffer flush before shutdown...")
        try:
            self.flush_buffer()
        except Exception as e:
            logger.error(f"Final flush error: {e}")
        logger.info("Flush thread terminated")

    def flush_buffer(self):
        """Flush write buffer to disk."""
        with self.buffer_lock:
            if not self.write_buffer:
                return

            to_flush = self.write_buffer.copy()
            self.write_buffer.clear()

        # Write all buffered memories
        for memory, compression_type in to_flush:
            self._save_memory_immediate(memory, compression_type)

    def shutdown(self):
        """FIX 629: Graceful shutdown of persistence system."""
        logger.info("Shutting down MemoryPersistence...")

        # Signal shutdown to background thread
        self._shutdown_event.set()

        # Wait for flush thread to complete
        if self.flush_thread and self.flush_thread.is_alive():
            self.flush_thread.join(timeout=5.0)
            if self.flush_thread.is_alive():
                logger.warning("Flush thread did not terminate in time")

        # Ensure all pending writes are flushed
        try:
            self.flush_buffer()
        except Exception as e:
            logger.error(f"Shutdown flush error: {e}")

        # Clean up ephemeral key file if it exists
        if ENCRYPTION_AVAILABLE:
            allow_ephemeral = (
                os.getenv("ALLOW_EPHEMERAL_KEY", "false").lower() == "true"
            )
            if allow_ephemeral:
                key_file = self.base_path / ".ephemeral_key"
                if key_file.exists():
                    try:
                        key_file.unlink()
                        logger.debug("Removed ephemeral key file")
                    except Exception as e:
                        logger.debug(
                            f"{self.__class__.__name__ if hasattr(self, '__class__') else 'Operation'} error: {e}"
                        )

        logger.info("MemoryPersistence shutdown complete")

    def save_memory(
        self,
        memory: Memory,
        compress: bool = True,
        compression_type: CompressionType = CompressionType.LZ4,
        immediate: bool = False,
    ) -> bool:
        """Save memory to disk with buffering."""
        try:
            # Validate input
            if not isinstance(memory, Memory):
                logger.error(f"Invalid memory object type: {type(memory)}")
                return False

            # Create version
            try:
                version_id = self.version_control.create_version(memory)
                memory.metadata["version_id"] = version_id
            except Exception as e:
                logger.warning(f"Failed to create version for {memory.id}: {e}")
                # Continue without versioning

            if immediate or len(self.write_buffer) >= self.buffer_size:
                # Write immediately
                self.flush_buffer()
                return self._save_memory_immediate(
                    memory, compression_type if compress else CompressionType.NONE
                )
            else:
                # Add to buffer
                with self.buffer_lock:
                    self.write_buffer.append(
                        (
                            copy.deepcopy(memory),
                            compression_type if compress else CompressionType.NONE,
                        )
                    )
                return True

        except Exception as e:
            logger.error(
                f"Failed to save memory {getattr(memory, 'id', 'unknown')}: {e}"
            )
            return False

    def _save_memory_immediate(
        self, memory: Memory, compression_type: CompressionType
    ) -> bool:
        """Save memory immediately to disk with Windows-compatible atomic writes."""
        try:
            # Prepare data: compress then encrypt
            if compression_type != CompressionType.NONE:
                data = self.compressor.compress(memory, compression_type)
                memory.compressed = True
                memory.compression_type = compression_type
            else:
                data = pickle.dumps(memory)

            # Encrypt if cipher is available
            if self.cipher:
                data = self.cipher.encrypt(data)

            # Get file paths
            file_path = self._get_memory_path(memory.id)

            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Use different strategy for Windows vs Unix
            import platform

            if platform.system() == "Windows":
                # Windows: Direct write with retry (fsync not reliable on Windows)
                success = self._atomic_write_windows(file_path, data, memory.id)
            else:
                # Unix: Proper atomic write with fsync
                success = self._atomic_write_unix(file_path, data, memory.id)

            if not success:
                return False

            # Save metadata
            self._save_metadata(memory, memory.metadata.get("version_id", "unknown"))

            # Update stats
            self.stats["total_saves"] += 1
            if compression_type != CompressionType.NONE:
                ratio = len(pickle.dumps(memory)) / max(1, len(data))
                self.stats["compression_ratio_avg"] = (
                    self.stats["compression_ratio_avg"] * 0.9 + ratio * 0.1
                )

            return True

        except Exception as e:
            logger.error(f"Failed to save memory {memory.id}: {e}", exc_info=True)
            return False

    def _atomic_write_windows(
        self, file_path: Path, data: bytes, memory_id: str
    ) -> bool:
        """Atomic write for Windows systems."""
        import tempfile

        try:
            # Write to temp file in same directory (important for atomic move)
            temp_fd, temp_path = tempfile.mkstemp(
                dir=file_path.parent, prefix=".tmp_", suffix=".mem"
            )
            temp_path = Path(temp_path)

            try:
                # Write data
                with os.fdopen(temp_fd, "wb") as f:
                    f.write(data)
                    f.flush()

                # Backup existing file if present
                backup_path = None
                if file_path.exists():
                    backup_path = file_path.with_suffix(".bak")
                    try:
                        if backup_path.exists():
                            backup_path.unlink()
                        shutil.copy2(file_path, backup_path)
                    except Exception as e:
                        logger.warning(f"Failed to create backup: {e}")

                # Atomic move (replace)
                shutil.move(str(temp_path), str(file_path))

                # Remove backup on success
                if backup_path and backup_path.exists():
                    try:
                        backup_path.unlink()
                    except Exception as e:
                        logger.debug(
                            f"{self.__class__.__name__ if hasattr(self, '__class__') else 'Operation'} error: {e}"
                        )

                return True

            except Exception as e:
                logger.error(f"Write failed: {e}")
                # Clean up temp file
                try:
                    if temp_path.exists():
                        temp_path.unlink()
                except Exception as e:
                    logger.debug(
                        f"{self.__class__.__name__ if hasattr(self, '__class__') else 'Operation'} error: {e}"
                    )
                return False

        except Exception as e:
            logger.error(f"Atomic write failed for memory {memory_id}: {e}")
            return False

    def _atomic_write_unix(self, file_path: Path, data: bytes, memory_id: str) -> bool:
        """Atomic write for Unix systems with fsync."""
        temp_path = file_path.with_suffix(file_path.suffix + ".tmp")
        backup_path = file_path.with_suffix(file_path.suffix + ".bak")

        try:
            # Create backup of existing file
            if file_path.exists():
                shutil.copy2(file_path, backup_path)

            # Write to temp file with fsync
            with open(temp_path, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())

            # Atomic rename
            temp_path.replace(file_path)

            # Fsync directory for durability
            try:
                dir_fd = os.open(file_path.parent, os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except (OSError, AttributeError):
                # Not all systems support directory fsync
                pass

            # Remove backup on success
            if backup_path.exists():
                backup_path.unlink()

            return True

        except Exception as e:
            logger.error(f"Atomic write failed for memory {memory_id}: {e}")

            # Restore from backup
            if backup_path.exists():
                try:
                    shutil.copy2(backup_path, file_path)
                    logger.info(f"Restored memory {memory_id} from backup")
                except Exception as e:
                    logger.debug(
                        f"{self.__class__.__name__ if hasattr(self, '__class__') else 'Operation'} error: {e}"
                    )
                try:
                    backup_path.unlink()
                except Exception as e:
                    logger.debug(
                        f"{self.__class__.__name__ if hasattr(self, '__class__') else 'Operation'} error: {e}"
                    )

            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception as e:
                    logger.debug(
                        f"{self.__class__.__name__ if hasattr(self, '__class__') else 'Operation'} error: {e}"
                    )

            return False

    def load_memory(
        self, memory_id: str, version_id: Optional[str] = None
    ) -> Optional[Memory]:
        """Load memory from disk, optionally at specific version."""
        try:
            # Check if specific version requested
            if version_id:
                version = self.version_control.get_version(memory_id, version_id)
                if not version:
                    return None

                # Load version file if it exists
                version_path = self.base_path / "versions" / f"{version_id}.mem"
                if version_path.exists():
                    file_path = version_path
                else:
                    file_path = self._get_memory_path(memory_id)
            else:
                file_path = self._get_memory_path(memory_id)

            if not file_path.exists():
                return None

            # Load data from disk
            with open(file_path, "rb") as f:
                data = f.read()

            # Decrypt if cipher is available
            if self.cipher:
                try:
                    data = self.cipher.decrypt(data)
                except Exception as e:
                    logger.error(
                        f"Failed to decrypt memory {memory_id}: {e}. Incorrect key or corrupted data."
                    )
                    return None

            # Load metadata
            metadata = self._load_metadata(memory_id)

            # Decompress if needed
            if metadata and metadata.get("compressed"):
                compression_type = CompressionType(
                    metadata.get("compression_type", "lz4")
                )
                content = self.compressor.decompress(data, compression_type)

                # Reconstruct memory
                memory = Memory(
                    id=memory_id,
                    type=MemoryType(metadata["type"]),
                    content=content,
                    timestamp=metadata["timestamp"],
                    importance=metadata.get("importance", 0.5),
                    metadata=metadata.get("metadata", {}),
                )
            else:
                memory = pickle.loads(data)

            # Update stats
            self.stats["total_loads"] += 1

            return memory

        except Exception as e:
            logger.error(f"Failed to load memory {memory_id}: {e}")
            return None

    def save_batch(
        self,
        memories: List[Memory],
        compress: bool = True,
        compression_type: CompressionType = CompressionType.LZ4,
        immediate: bool = False,
    ) -> int:
        """Save batch of memories efficiently."""
        if immediate:
            # Force immediate flush
            saved_count = 0
            for memory in memories:
                if self._save_memory_immediate(
                    memory, compression_type if compress else CompressionType.NONE
                ):
                    saved_count += 1
            return saved_count

        # Use buffering
        saved_count = 0
        with self.buffer_lock:
            for memory in memories:
                self.write_buffer.append(
                    (
                        copy.deepcopy(memory),
                        compression_type if compress else CompressionType.NONE,
                    )
                )
                saved_count += 1

        # Flush if buffer is full
        if len(self.write_buffer) >= self.buffer_size * 2:
            self.flush_buffer()

        return saved_count

    def load_batch(self, memory_ids: List[str]) -> List[Memory]:
        """Load batch of memories with parallel loading."""
        memories = []

        # Use thread pool for parallel loading
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(self.load_memory, mid) for mid in memory_ids]

            for future in futures:
                memory = future.result()
                if memory:
                    memories.append(memory)

        return memories

    def delete_memory(self, memory_id: str, permanent: bool = False) -> bool:
        """Delete memory from disk."""
        try:
            if not permanent:
                # Move to trash instead of deleting
                trash_path = self.base_path / "trash"
                trash_path.mkdir(exist_ok=True)

                file_path = self._get_memory_path(memory_id)
                if file_path.exists():
                    trash_file = trash_path / f"{memory_id}_{int(time.time())}.mem"
                    shutil.move(str(file_path), str(trash_file))
            else:
                # Permanent deletion
                file_path = self._get_memory_path(memory_id)
                if file_path.exists():
                    file_path.unlink()

                # Delete metadata
                metadata_path = self._get_metadata_path(memory_id)
                if metadata_path.exists():
                    metadata_path.unlink()

            return True

        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False

    def list_memories(self, pattern: Optional[str] = None) -> List[str]:
        """List all persisted memory IDs with optional pattern matching."""
        memory_ids = []

        if pattern:
            # Search in subdirectories for pattern
            glob_pattern = f"**/{pattern}*.mem"
        else:
            # Get all memories from all subdirectories
            glob_pattern = "**/*.mem"

        for file_path in self.memories_path.glob(glob_pattern):
            memory_id = file_path.stem
            memory_ids.append(memory_id)

        return memory_ids

    def checkpoint(
        self, memories: Dict[str, Memory], name: Optional[str] = None
    ) -> bool:
        """Create named checkpoint of all memories."""
        try:
            # Flush buffer first
            self.flush_buffer()

            # Generate checkpoint name
            if name:
                checkpoint_name = f"checkpoint_{name}_{int(time.time())}"
            else:
                checkpoint_name = f"checkpoint_{int(time.time())}"

            checkpoint_path = self.base_path / f"{checkpoint_name}.pkl"

            # Prepare checkpoint data
            checkpoint_data = {
                "name": checkpoint_name,
                "timestamp": time.time(),
                "memory_count": len(memories),
                "version": "1.0",
                "memories": {},
            }

            # FIX: Save entire Memory objects, not just content
            for memory_id, memory in memories.items():
                # Verify we have a Memory object
                if not isinstance(memory, Memory):
                    logger.error(
                        f"Checkpoint requires Memory objects, got {type(memory)} for {memory_id}"
                    )
                    continue

                # Pickle the ENTIRE Memory object
                serialized_memory = pickle.dumps(
                    memory, protocol=pickle.HIGHEST_PROTOCOL
                )

                # Then compress it
                compressed = lz4.frame.compress(serialized_memory)

                checkpoint_data["memories"][memory_id] = compressed

            # Serialize and encrypt checkpoint
            serialized_data = pickle.dumps(
                checkpoint_data, protocol=pickle.HIGHEST_PROTOCOL
            )
            if self.cipher:
                serialized_data = self.cipher.encrypt(serialized_data)

            # Save checkpoint atomically
            temp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
            with open(temp_path, "wb") as f:
                f.write(serialized_data)
            temp_path.replace(checkpoint_path)

            self.last_checkpoint = time.time()
            logger.info(
                f"Created checkpoint '{checkpoint_name}' with {len(checkpoint_data['memories'])} memories"
            )

            # Create backup
            self._create_backup(checkpoint_path)

            return True

        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
            return False

    def restore_from_checkpoint(
        self, checkpoint_file: Optional[str] = None
    ) -> Dict[str, Memory]:
        """Restore memories from checkpoint."""
        try:
            if checkpoint_file is None:
                # Find latest checkpoint
                checkpoints = list(self.base_path.glob("checkpoint_*.pkl"))
                if not checkpoints:
                    # Check backups
                    checkpoints = list(self.backups_path.glob("checkpoint_*.pkl"))
                    if not checkpoints:
                        return {}

                checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
            else:
                checkpoint_path = self.base_path / checkpoint_file
                if not checkpoint_path.exists():
                    # Check in backups
                    checkpoint_path = self.backups_path / checkpoint_file
                    if not checkpoint_path.exists():
                        return {}

            # Load and decrypt checkpoint
            with open(checkpoint_path, "rb") as f:
                data = f.read()

            if self.cipher:
                try:
                    data = self.cipher.decrypt(data)
                except Exception as e:
                    logger.error(
                        f"Failed to decrypt checkpoint {checkpoint_path}. Trying as unencrypted. Error: {e}"
                    )

            checkpoint_data = pickle.loads(data)

            # FIX: Decompress to get complete Memory objects
            memories = {}
            for memory_id, compressed_data in checkpoint_data["memories"].items():
                try:
                    # Decompress the LZ4 data
                    decompressed_bytes = lz4.frame.decompress(compressed_data)

                    # Unpickle to get the Memory object
                    memory = pickle.loads(decompressed_bytes)

                    # Verify it's actually a Memory object
                    if isinstance(memory, Memory):
                        memories[memory_id] = memory
                    else:
                        logger.error(
                            f"Checkpoint contained non-Memory object for {memory_id}: {type(memory)}"
                        )

                except Exception as e:
                    logger.error(f"Failed to restore memory {memory_id}: {e}")
                    continue

            logger.info(f"Restored {len(memories)} memories from checkpoint")
            return memories

        except Exception as e:
            logger.error(f"Failed to restore from checkpoint: {e}")
            return {}

    def _create_backup(self, source_path: Path):
        """Create backup of file."""
        try:
            backup_name = (
                f"{source_path.stem}_backup_{int(time.time())}{source_path.suffix}"
            )
            backup_path = self.backups_path / backup_name
            shutil.copy2(str(source_path), str(backup_path))

            self.stats["last_backup"] = time.time()

            # Clean old backups (keep last 10)
            backups = sorted(
                self.backups_path.glob("*_backup_*"), key=lambda p: p.stat().st_mtime
            )
            if len(backups) > 10:
                for old_backup in backups[:-10]:
                    old_backup.unlink()

        except Exception as e:
            logger.warning(f"Failed to create backup: {e}")

    def _get_memory_path(self, memory_id: str) -> Path:
        """Get file path for memory with robust subdirectory handling."""
        # Ensure memory_id is valid
        if not memory_id or len(memory_id) < 3:
            # Don't use subdirs for very short IDs
            return self.memories_path / f"{memory_id}.mem"

        # Use first 2 chars as subdirectory (ensure alphanumeric)
        subdir = "".join(c for c in memory_id[:2] if c.isalnum())
        if not subdir:
            subdir = "misc"

        subdir_path = self.memories_path / subdir

        # Create subdirectory with proper error handling
        try:
            subdir_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(
                f"Failed to create subdir {subdir}: {e}, using flat structure"
            )
            return self.memories_path / f"{memory_id}.mem"

        return subdir_path / f"{memory_id}.mem"

    def _get_metadata_path(self, memory_id: str) -> Path:
        """Get metadata file path with robust subdirectory handling."""
        if not memory_id or len(memory_id) < 3:
            return self.metadata_path / f"{memory_id}.json"

        subdir = "".join(c for c in memory_id[:2] if c.isalnum())
        if not subdir:
            subdir = "misc"

        subdir_path = self.metadata_path / subdir

        try:
            subdir_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Failed to create metadata subdir: {e}")
            return self.metadata_path / f"{memory_id}.json"

        return subdir_path / f"{memory_id}.json"

    def _save_metadata(self, memory: Memory, version_id: str):
        """Save memory metadata."""
        metadata = {
            "id": memory.id,
            "type": memory.type.value,
            "timestamp": memory.timestamp,
            "importance": memory.importance,
            "compressed": memory.compressed,
            "compression_type": memory.compression_type.value
            if memory.compression_type
            else None,
            "version_id": version_id,
            "metadata": memory.metadata,
            "last_modified": time.time(),
        }

        metadata_path = self._get_metadata_path(memory.id)

        # Ensure parent directory exists
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        temp_path = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
        with open(temp_path, "w") as f:
            json.dump(metadata, f, indent=2)
        temp_path.replace(metadata_path)

    def _load_metadata(self, memory_id: str) -> Optional[Dict]:
        """Load memory metadata."""
        metadata_path = self._get_metadata_path(memory_id)

        if not metadata_path.exists():
            return None

        with open(metadata_path, "r") as f:
            return json.load(f)

    def cleanup_old_versions(self, max_age_days: int = 30) -> int:
        """Clean up old memory versions."""
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 3600

        cleaned = 0

        # Clean version files
        versions_path = self.base_path / "versions"
        if versions_path.exists():
            for version_file in versions_path.glob("*.mem"):
                if current_time - version_file.stat().st_mtime > max_age_seconds:
                    version_file.unlink()
                    cleaned += 1

        # Clean trash
        trash_path = self.base_path / "trash"
        if trash_path.exists():
            for trash_file in trash_path.glob("*.mem"):
                if current_time - trash_file.stat().st_mtime > max_age_seconds:
                    trash_file.unlink()
                    cleaned += 1

        logger.info(f"Cleaned up {cleaned} old files")
        return cleaned

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        total_size = 0
        file_count = 0

        for path in [
            self.memories_path,
            self.metadata_path,
            self.base_path / "versions",
            self.backups_path,
        ]:
            if path.exists():
                for file in path.rglob("*"):
                    if file.is_file():
                        total_size += file.stat().st_size
                        file_count += 1

        return {
            "total_files": file_count,
            "total_size_mb": total_size / (1024 * 1024),
            "memories_count": len(list(self.memories_path.glob("**/*.mem"))),
            "checkpoints_count": len(list(self.base_path.glob("checkpoint_*.pkl"))),
            "backups_count": len(list(self.backups_path.glob("*"))),
            **self.stats,
        }
