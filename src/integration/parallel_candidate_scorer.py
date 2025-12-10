"""
Vulcan LLM - Advanced Parallel Candidate Scorer
================================================

A comprehensive, production-ready candidate scoring system for speculative decoding
and LLM inference optimization. Features multiple embedding architectures, advanced
scoring metrics, caching, multi-GPU support, and extensive configurability.

Author: Vulcan AI Research Team
Version: 2.0.0
License: MIT
"""

import asyncio
import hashlib
import json
import logging
import math
import pickle
import threading
import time
import warnings
from collections import OrderedDict, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from functools import lru_cache, wraps
from pathlib import Path
from typing import (Any, Callable, Dict, Generic, List, NamedTuple, Optional,
                    Protocol, Sequence, Set, Tuple, TypeVar, Union)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .security_fixes import safe_pickle_load

# ==================== LOGGING CONFIGURATION ====================


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Setup logger with consistent formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


log = setup_logger(__name__)

# ==================== TYPE DEFINITIONS ====================

T = TypeVar("T")
ScoreResult = Tuple[float, Dict[str, Any]]
CandidateType = Union[str, List[int], List[str], int]


class ScoringStrategy(Enum):
    """Available scoring strategies."""

    SEMANTIC = auto()
    STATISTICAL = auto()
    HYBRID = auto()
    ENSEMBLE = auto()
    ADAPTIVE = auto()


class EmbeddingArchitecture(Enum):
    """Supported embedding architectures."""

    LSTM = auto()
    BILSTM = auto()
    GRU = auto()
    TRANSFORMER = auto()
    CONV1D = auto()
    HYBRID_LSTM_ATTENTION = auto()


class PenaltyType(Enum):
    """Types of penalties."""

    LENGTH = auto()
    DIVERSITY = auto()
    COHERENCE = auto()
    OOV = auto()
    RISK = auto()
    REPETITION = auto()
    FLUENCY = auto()
    PERPLEXITY = auto()


class DeviceType(Enum):
    """Device types for computation."""

    CPU = auto()
    CUDA = auto()
    MPS = auto()


# ==================== CONFIGURATION CLASSES ====================


@dataclass
class DeviceConfig:
    """Device and parallelization configuration."""

    device_type: DeviceType = DeviceType.CPU
    cuda_device_id: int = 0
    enable_multi_gpu: bool = False
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    enable_mixed_precision: bool = False
    enable_amp: bool = False
    use_tf32: bool = True
    cudnn_benchmark: bool = True

    def get_device(self) -> torch.device:
        """Get the primary torch device."""
        if self.device_type == DeviceType.CUDA and torch.cuda.is_available():
            return torch.device(f"cuda:{self.cuda_device_id}")
        elif self.device_type == DeviceType.MPS and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def get_all_devices(self) -> List[torch.device]:
        """Get all available devices for multi-GPU."""
        if self.enable_multi_gpu and torch.cuda.is_available():
            return [torch.device(f"cuda:{i}") for i in self.gpu_ids]
        return [self.get_device()]


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""

    architecture: EmbeddingArchitecture = EmbeddingArchitecture.HYBRID_LSTM_ATTENTION
    vocab_size: int = 50000
    embedding_dim: int = 384
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    bidirectional: bool = True

    # Transformer-specific
    num_attention_heads: int = 8
    num_transformer_layers: int = 4
    feedforward_dim: int = 1024

    # Convolution-specific
    kernel_sizes: List[int] = field(default_factory=lambda: [3, 5, 7])
    num_filters: int = 128

    # Advanced options
    use_pretrained: bool = False
    pretrained_model_path: Optional[str] = None
    freeze_embeddings: bool = False
    enable_layer_norm: bool = True
    enable_residual: bool = True

    # Quantization
    enable_quantization: bool = False
    quantization_bits: int = 8


@dataclass
class PenaltyConfig:
    """Configuration for penalty functions."""

    # Length penalties
    length_penalty_factor: float = 0.01
    length_penalty_curve: str = "linear"  # linear, quadratic, logarithmic, exponential
    max_length_threshold: int = 1000
    min_length_threshold: int = 1

    # Diversity penalties
    diversity_penalty_factor: float = 0.5
    diversity_window_size: int = 50
    diversity_threshold: float = 0.85
    enable_semantic_clustering: bool = True
    cluster_merge_threshold: float = 0.95

    # Coherence penalties
    coherence_penalty_factor: float = 0.02
    coherence_window_size: int = 20
    enable_perplexity_penalty: bool = True
    perplexity_penalty_factor: float = 0.1

    # OOV penalties
    oov_penalty_value: float = 0.5
    oov_penalty_curve: str = "linear"
    enable_subword_fallback: bool = True

    # Risk penalties
    risk_penalty_value: float = 0.8
    risk_blacklist: Set[str] = field(
        default_factory=lambda: {
            "profane",
            "error",
            "block",
            "spam",
            "toxic",
            "hate",
            "violence",
            "nsfw",
            "malicious",
            "exploit",
        }
    )
    enable_advanced_safety: bool = True
    safety_model_threshold: float = 0.7

    # Repetition penalties
    repetition_penalty_factor: float = 0.3
    repetition_window_size: int = 30
    ngram_sizes: List[int] = field(default_factory=lambda: [2, 3, 4])

    # Fluency penalties
    fluency_penalty_factor: float = 0.15
    enable_grammar_check: bool = False

    # Adaptive penalties
    enable_adaptive_penalties: bool = True
    adaptation_rate: float = 0.1
    context_sensitivity: float = 0.5


@dataclass
class CacheConfig:
    """Configuration for caching system."""

    enable_cache: bool = True
    max_cache_size: int = 10000
    cache_ttl_seconds: float = 3600.0
    enable_persistent_cache: bool = False
    cache_directory: str = "./cache/embeddings"
    eviction_policy: str = "lru"  # lru, lfu, fifo
    enable_compression: bool = True
    compression_level: int = 6


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""

    enable_profiling: bool = False
    profiling_interval: int = 100

    # Parallel execution
    max_workers: int = 4
    use_process_pool: bool = False
    batch_size: int = 32
    enable_batch_optimization: bool = True

    # Async configuration
    parallel_timeout_seconds: float = 2.0
    enable_streaming: bool = False

    # Memory management
    enable_gradient_checkpointing: bool = False
    max_memory_gb: float = 8.0
    garbage_collection_threshold: int = 1000


@dataclass
class ScoringConfig:
    """Main scoring configuration."""

    strategy: ScoringStrategy = ScoringStrategy.HYBRID

    # Weight configuration for hybrid/ensemble
    semantic_weight: float = 0.6
    statistical_weight: float = 0.2
    fluency_weight: float = 0.2

    # Ensemble configuration
    ensemble_methods: List[str] = field(
        default_factory=lambda: ["semantic", "statistical", "ngram"]
    )
    ensemble_aggregation: str = "weighted_average"  # weighted_average, max, min, median

    # Score normalization
    enable_normalization: bool = True
    normalization_method: str = "minmax"  # minmax, zscore, softmax, sigmoid
    score_clip_range: Tuple[float, float] = (-1.0, 1.0)

    # Advanced features
    enable_uncertainty_estimation: bool = True
    enable_explainability: bool = True
    explainability_top_k: int = 5

    # A/B Testing
    enable_ab_testing: bool = False
    ab_test_split: float = 0.5
    ab_test_variant: str = "A"


@dataclass
class VulcanScorerConfig:
    """Master configuration for the Vulcan Scorer system."""

    device: DeviceConfig = field(default_factory=DeviceConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    penalty: PenaltyConfig = field(default_factory=PenaltyConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)

    # Global settings
    random_seed: int = 42
    deterministic: bool = False
    verbose: bool = False
    log_level: str = "INFO"

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
        self._setup_logging()
        self._set_random_seeds()

    def _validate_config(self):
        """Validate configuration parameters."""
        assert 0.0 <= self.scoring.semantic_weight <= 1.0
        assert 0.0 <= self.scoring.statistical_weight <= 1.0
        assert 0.0 <= self.scoring.fluency_weight <= 1.0
        assert self.embedding.embedding_dim > 0
        assert self.embedding.vocab_size > 0
        assert self.performance.max_workers > 0
        assert self.performance.batch_size > 0

    def _setup_logging(self):
        """Setup logging based on configuration."""
        level = getattr(logging, self.log_level.upper(), logging.INFO)
        log.setLevel(level)

    def _set_random_seeds(self):
        """Set random seeds for reproducibility."""
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        if self.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def save(self, path: str):
        """Save configuration to file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)
        log.info(f"Configuration saved to {path}")

    @classmethod
    def load(cls, path: str) -> "VulcanScorerConfig":
        """Load configuration from file."""
        with open(path, "r") as f:
            data = json.load(f)
        # Reconstruct nested dataclasses
        config = cls(
            device=DeviceConfig(**data.get("device", {})),
            embedding=EmbeddingConfig(**data.get("embedding", {})),
            penalty=PenaltyConfig(**data.get("penalty", {})),
            cache=CacheConfig(**data.get("cache", {})),
            performance=PerformanceConfig(**data.get("performance", {})),
            scoring=ScoringConfig(**data.get("scoring", {})),
        )
        log.info(f"Configuration loaded from {path}")
        return config


# ==================== TOKENIZATION SYSTEM ====================


class TokenizerProtocol(Protocol):
    """Protocol for tokenizer implementations."""

    def encode(self, text: str) -> List[int]: ...
    def decode(self, tokens: List[int]) -> str: ...
    def get_vocab_size(self) -> int: ...


class SimpleTokenizer:
    """Simple word-based tokenizer with vocabulary management."""

    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        self.word_to_id: Dict[str, int] = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3,
        }
        self.id_to_word: Dict[int, str] = {v: k for k, v in self.word_to_id.items()}
        self.next_id = 4
        self._lock = threading.Lock()

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        words = text.lower().split()
        tokens = []
        for word in words:
            if word not in self.word_to_id:
                with self._lock:
                    if word not in self.word_to_id and self.next_id < self.vocab_size:
                        self.word_to_id[word] = self.next_id
                        self.id_to_word[self.next_id] = word
                        self.next_id += 1
            tokens.append(self.word_to_id.get(word, 1))  # 1 = <UNK>
        return tokens

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text."""
        words = [self.id_to_word.get(t, "<UNK>") for t in tokens]
        return " ".join(words)

    def get_vocab_size(self) -> int:
        """Get current vocabulary size."""
        return len(self.word_to_id)


class BPETokenizer:
    """Byte-Pair Encoding tokenizer (simplified implementation)."""

    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        self.merges: Dict[Tuple[str, str], str] = {}
        self.vocab: Dict[str, int] = {}
        self._initialize_vocab()

    def _initialize_vocab(self):
        """Initialize basic vocabulary with characters."""
        self.vocab = {chr(i): i for i in range(256)}
        self.vocab.update({"<PAD>": 256, "<UNK>": 257, "<BOS>": 258, "<EOS>": 259})

    def encode(self, text: str) -> List[int]:
        """Encode text using BPE."""
        # Simplified BPE encoding
        words = text.split()
        tokens = []
        for word in words:
            word_tokens = [self.vocab.get(c, 257) for c in word]
            tokens.extend(word_tokens)
        return tokens

    def decode(self, tokens: List[int]) -> str:
        """Decode BPE tokens."""
        id_to_char = {v: k for k, v in self.vocab.items() if v < 256}
        chars = [id_to_char.get(t, "?") for t in tokens]
        return "".join(chars)

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)


# ==================== CACHING SYSTEM ====================


class EmbeddingCache:
    """Thread-safe LRU cache for embeddings with optional persistence."""

    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache: OrderedDict[str, Tuple[torch.Tensor, float]] = OrderedDict()
        self.hit_count = 0
        self.miss_count = 0
        self._lock = threading.Lock()

        if config.enable_persistent_cache:
            self.cache_dir = Path(config.cache_directory)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _make_key(self, text: Union[str, List[int]]) -> str:
        """Create cache key from text or tokens."""
        if isinstance(text, str):
            data = text.encode("utf-8")
        else:
            data = str(text).encode("utf-8")
        return hashlib.sha256(data).hexdigest()

    def get(self, key_data: Union[str, List[int]]) -> Optional[torch.Tensor]:
        """Get embedding from cache."""
        if not self.config.enable_cache:
            return None

        key = self._make_key(key_data)
        current_time = time.time()

        with self._lock:
            if key in self.cache:
                embedding, timestamp = self.cache[key]

                # Check TTL
                if current_time - timestamp < self.config.cache_ttl_seconds:
                    # Move to end (LRU)
                    self.cache.move_to_end(key)
                    self.hit_count += 1
                    return embedding.clone()
                else:
                    # Expired
                    del self.cache[key]

            self.miss_count += 1
            return None

    def put(self, key_data: Union[str, List[int]], embedding: torch.Tensor):
        """Put embedding in cache."""
        if not self.config.enable_cache:
            return

        key = self._make_key(key_data)
        current_time = time.time()

        with self._lock:
            # Evict if necessary
            while len(self.cache) >= self.config.max_cache_size:
                if self.config.eviction_policy == "lru":
                    self.cache.popitem(last=False)
                elif self.config.eviction_policy == "fifo":
                    self.cache.popitem(last=False)
                else:  # Default to LRU
                    self.cache.popitem(last=False)

            self.cache[key] = (embedding.clone(), current_time)

    def clear(self):
        """Clear the cache."""
        with self._lock:
            self.cache.clear()
            self.hit_count = 0
            self.miss_count = 0
        log.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total if total > 0 else 0.0
        return {
            "size": len(self.cache),
            "max_size": self.config.max_cache_size,
            "hits": self.hit_count,
            "misses": self.miss_count,
            "hit_rate": hit_rate,
        }


# ==================== EMBEDDING ARCHITECTURES ====================


class BasedEmbedder(nn.Module):
    """Base class for all embedding architectures."""

    def __init__(self, config: EmbeddingConfig, device: torch.device):
        super().__init__()
        self.config = config
        self.device = device
        self.embedding = nn.Embedding(
            config.vocab_size, config.embedding_dim, padding_idx=0
        )

        if config.freeze_embeddings:
            self.embedding.weight.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - to be implemented by subclasses."""
        raise NotImplementedError

    @torch.no_grad()
    def encode(
        self, sequences: Union[str, List[Union[str, List[int]]], List[int]]
    ) -> torch.Tensor:
        """Encode sequences to embeddings."""
        raise NotImplementedError


class LSTMEmbedder(BasedEmbedder):
    """LSTM-based sequence embedder."""

    def __init__(self, config: EmbeddingConfig, device: torch.device):
        super().__init__(config, device)

        self.lstm = nn.LSTM(
            config.embedding_dim,
            config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional,
        )

        lstm_output_size = config.hidden_size * (2 if config.bidirectional else 1)
        self.projection = nn.Linear(lstm_output_size, config.embedding_dim)

        if config.enable_layer_norm:
            self.layer_norm = nn.LayerNorm(config.embedding_dim)
        else:
            self.layer_norm = None

        self._initialize_weights()
        self.to(device)

    def _initialize_weights(self):
        """Initialize model weights."""
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LSTM."""
        # x: [batch, seq_len]
        embedded = self.embedding(x)  # [batch, seq_len, embed_dim]

        lstm_out, (h_n, c_n) = self.lstm(embedded)
        # lstm_out: [batch, seq_len, hidden_size * num_directions]

        # Average pooling over sequence
        pooled = lstm_out.mean(dim=1)  # [batch, hidden_size * num_directions]

        # Project back to embedding dimension
        output = self.projection(pooled)  # [batch, embed_dim]

        if self.layer_norm is not None:
            output = self.layer_norm(output)

        # L2 normalization
        output = F.normalize(output, p=2, dim=1)

        return output

    @torch.no_grad()
    def encode(
        self, sequences: Union[str, List[Union[str, List[int]]], List[int]]
    ) -> torch.Tensor:
        """Encode sequences to embeddings."""
        self.eval()

        if isinstance(sequences, (str, list)) and not isinstance(sequences, List[int]):
            if isinstance(sequences, str):
                sequences = [sequences]

            # Convert to token IDs
            token_ids_list = []
            for seq in sequences:
                if isinstance(seq, str):
                    # Simple tokenization
                    ids = [
                        min(
                            sum(ord(c) for c in w) % (self.config.vocab_size - 10),
                            self.config.vocab_size - 1,
                        )
                        for w in seq.split()
                    ]
                elif isinstance(seq, list):
                    ids = [int(t) for t in seq if isinstance(t, int)]
                else:
                    ids = [0]
                token_ids_list.append(torch.tensor(ids, dtype=torch.long))

            # Pad sequences
            padded = nn.utils.rnn.pad_sequence(
                token_ids_list, batch_first=True, padding_value=0
            )
            padded = padded.to(self.device)
        else:
            # Already token IDs
            if isinstance(sequences, list):
                sequences = torch.tensor(sequences, dtype=torch.long).unsqueeze(0)
            padded = sequences.to(self.device)

        return self.forward(padded)


class TransformerEmbedder(BasedEmbedder):
    """Transformer-based sequence embedder."""

    def __init__(self, config: EmbeddingConfig, device: torch.device):
        super().__init__(config, device)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            config.embedding_dim, max_len=5000, dropout=config.dropout
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.num_attention_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_transformer_layers
        )

        if config.enable_layer_norm:
            self.layer_norm = nn.LayerNorm(config.embedding_dim)
        else:
            self.layer_norm = None

        self._initialize_weights()
        self.to(device)

    def _initialize_weights(self):
        """Initialize model weights."""
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through Transformer."""
        # x: [batch, seq_len]
        embedded = self.embedding(x)  # [batch, seq_len, embed_dim]
        embedded = self.positional_encoding(embedded)

        # Create padding mask
        if mask is None:
            mask = x == 0  # Padding mask

        transformer_out = self.transformer(embedded, src_key_padding_mask=mask)
        # transformer_out: [batch, seq_len, embed_dim]

        # Mean pooling (excluding padding)
        if mask is not None:
            mask_expanded = (~mask).unsqueeze(-1).float()
            pooled = (transformer_out * mask_expanded).sum(dim=1) / mask_expanded.sum(
                dim=1
            ).clamp(min=1)
        else:
            pooled = transformer_out.mean(dim=1)

        if self.layer_norm is not None:
            pooled = self.layer_norm(pooled)

        # L2 normalization
        output = F.normalize(pooled, p=2, dim=1)

        return output

    @torch.no_grad()
    def encode(
        self, sequences: Union[str, List[Union[str, List[int]]], List[int]]
    ) -> torch.Tensor:
        """Encode sequences to embeddings."""
        self.eval()

        if isinstance(sequences, (str, list)) and not isinstance(sequences, List[int]):
            if isinstance(sequences, str):
                sequences = [sequences]

            token_ids_list = []
            for seq in sequences:
                if isinstance(seq, str):
                    ids = [
                        min(
                            sum(ord(c) for c in w) % (self.config.vocab_size - 10),
                            self.config.vocab_size - 1,
                        )
                        for w in seq.split()
                    ]
                elif isinstance(seq, list):
                    ids = [int(t) for t in seq if isinstance(t, int)]
                else:
                    ids = [0]
                token_ids_list.append(torch.tensor(ids, dtype=torch.long))

            padded = nn.utils.rnn.pad_sequence(
                token_ids_list, batch_first=True, padding_value=0
            )
            padded = padded.to(self.device)
        else:
            if isinstance(sequences, list):
                sequences = torch.tensor(sequences, dtype=torch.long).unsqueeze(0)
            padded = sequences.to(self.device)

        return self.forward(padded)


class HybridLSTMAttentionEmbedder(BasedEmbedder):
    """Hybrid LSTM with self-attention mechanism."""

    def __init__(self, config: EmbeddingConfig, device: torch.device):
        super().__init__(config, device)

        # BiLSTM
        self.lstm = nn.LSTM(
            config.embedding_dim,
            config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=True,
        )

        lstm_output_size = config.hidden_size * 2  # Bidirectional

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_size,
            num_heads=config.num_attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )

        # Projection layers
        self.projection = nn.Sequential(
            nn.Linear(lstm_output_size, config.embedding_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embedding_dim, config.embedding_dim),
        )

        if config.enable_layer_norm:
            self.layer_norm1 = nn.LayerNorm(lstm_output_size)
            self.layer_norm2 = nn.LayerNorm(config.embedding_dim)
        else:
            self.layer_norm1 = None
            self.layer_norm2 = None

        self._initialize_weights()
        self.to(device)

    def _initialize_weights(self):
        """Initialize model weights."""
        nn.init.xavier_uniform_(self.embedding.weight)
        for module in self.projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through hybrid architecture."""
        # x: [batch, seq_len]
        embedded = self.embedding(x)  # [batch, seq_len, embed_dim]

        # LSTM encoding
        lstm_out, _ = self.lstm(embedded)  # [batch, seq_len, hidden*2]

        if self.layer_norm1 is not None:
            lstm_out = self.layer_norm1(lstm_out)

        # Self-attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        # attn_out: [batch, seq_len, hidden*2]

        # Residual connection
        if self.config.enable_residual:
            attn_out = attn_out + lstm_out

        # Average pooling
        pooled = attn_out.mean(dim=1)  # [batch, hidden*2]

        # Projection
        output = self.projection(pooled)  # [batch, embed_dim]

        if self.layer_norm2 is not None:
            output = self.layer_norm2(output)

        # L2 normalization
        output = F.normalize(output, p=2, dim=1)

        return output

    @torch.no_grad()
    def encode(
        self, sequences: Union[str, List[Union[str, List[int]]], List[int]]
    ) -> torch.Tensor:
        """Encode sequences to embeddings."""
        self.eval()

        if isinstance(sequences, (str, list)) and not isinstance(sequences, List[int]):
            if isinstance(sequences, str):
                sequences = [sequences]

            token_ids_list = []
            for seq in sequences:
                if isinstance(seq, str):
                    ids = [
                        min(
                            sum(ord(c) for c in w) % (self.config.vocab_size - 10),
                            self.config.vocab_size - 1,
                        )
                        for w in seq.split()
                    ]
                elif isinstance(seq, list):
                    ids = [int(t) for t in seq if isinstance(t, int)]
                else:
                    ids = [0]
                token_ids_list.append(torch.tensor(ids, dtype=torch.long))

            padded = nn.utils.rnn.pad_sequence(
                token_ids_list, batch_first=True, padding_value=0
            )
            padded = padded.to(self.device)
        else:
            if isinstance(sequences, list):
                sequences = torch.tensor(sequences, dtype=torch.long).unsqueeze(0)
            padded = sequences.to(self.device)

        return self.forward(padded)


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding."""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ==================== EMBEDDER FACTORY ====================


class EmbedderFactory:
    """Factory for creating embedding models."""

    @staticmethod
    def create(config: EmbeddingConfig, device: torch.device) -> BasedEmbedder:
        """Create embedder based on configuration."""
        arch = config.architecture

        if arch == EmbeddingArchitecture.LSTM:
            return LSTMEmbedder(config, device)
        elif arch == EmbeddingArchitecture.TRANSFORMER:
            return TransformerEmbedder(config, device)
        elif arch == EmbeddingArchitecture.HYBRID_LSTM_ATTENTION:
            return HybridLSTMAttentionEmbedder(config, device)
        else:
            log.warning(f"Architecture {arch} not fully implemented, using LSTM")
            return LSTMEmbedder(config, device)


# ==================== SIMILARITY METRICS ====================


class SimilarityCalculator:
    """Advanced similarity calculations."""

    @staticmethod
    @torch.no_grad()
    def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
        """Calculate cosine similarity between two vectors."""
        return F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()

    @staticmethod
    @torch.no_grad()
    def cosine_similarity_batch(a: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Calculate cosine similarity between one vector and a batch."""
        # a: [1, D] or [D], B: [N, D]
        if a.dim() == 1:
            a = a.unsqueeze(0)
        scores = B @ a.T  # [N, 1]
        return scores.squeeze(-1)  # [N]

    @staticmethod
    @torch.no_grad()
    def euclidean_distance(a: torch.Tensor, b: torch.Tensor) -> float:
        """Calculate Euclidean distance."""
        return torch.dist(a, b, p=2).item()

    @staticmethod
    @torch.no_grad()
    def manhattan_distance(a: torch.Tensor, b: torch.Tensor) -> float:
        """Calculate Manhattan distance."""
        return torch.dist(a, b, p=1).item()

    @staticmethod
    @torch.no_grad()
    def angular_distance(a: torch.Tensor, b: torch.Tensor) -> float:
        """Calculate angular distance."""
        cos_sim = F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
        return math.acos(max(-1.0, min(1.0, cos_sim))) / math.pi


# ==================== STATISTICAL FEATURES ====================


class StatisticalFeatureExtractor:
    """Extract statistical features from candidates."""

    @staticmethod
    def extract_features(candidate: str, context: Dict[str, Any]) -> Dict[str, float]:
        """Extract comprehensive statistical features."""
        features = {}

        words = candidate.split()
        features["word_count"] = len(words)
        features["char_count"] = len(candidate)
        features["avg_word_length"] = np.mean([len(w) for w in words]) if words else 0.0
        features["unique_word_ratio"] = len(set(words)) / len(words) if words else 0.0

        # Character-level features
        features["digit_ratio"] = (
            sum(c.isdigit() for c in candidate) / len(candidate) if candidate else 0.0
        )
        features["uppercase_ratio"] = (
            sum(c.isupper() for c in candidate) / len(candidate) if candidate else 0.0
        )
        features["punctuation_ratio"] = (
            sum(not c.isalnum() and not c.isspace() for c in candidate) / len(candidate)
            if candidate
            else 0.0
        )

        # Entropy
        if words:
            word_freq = defaultdict(int)
            for word in words:
                word_freq[word] += 1
            probs = np.array(list(word_freq.values())) / len(words)
            features["entropy"] = -np.sum(probs * np.log2(probs + 1e-10))
        else:
            features["entropy"] = 0.0

        return features


# ==================== PENALTY CALCULATORS ====================


class PenaltyCalculator:
    """Calculate various penalties for candidate scoring."""

    def __init__(self, config: PenaltyConfig):
        self.config = config
        self.ngram_cache: Dict[int, Set[Tuple]] = defaultdict(set)

    def calculate_length_penalty(
        self, candidate_length: int, context_length: int
    ) -> float:
        """Calculate length penalty with configurable curves."""
        if candidate_length < self.config.min_length_threshold:
            return self.config.length_penalty_factor * (
                self.config.min_length_threshold - candidate_length
            )

        if candidate_length > self.config.max_length_threshold:
            excess = candidate_length - self.config.max_length_threshold
            if self.config.length_penalty_curve == "quadratic":
                return self.config.length_penalty_factor * (excess**2)
            elif self.config.length_penalty_curve == "exponential":
                return self.config.length_penalty_factor * (math.exp(excess / 100) - 1)
            elif self.config.length_penalty_curve == "logarithmic":
                return self.config.length_penalty_factor * math.log1p(excess)
            else:  # linear
                return self.config.length_penalty_factor * excess

        return 0.0

    def calculate_coherence_penalty(
        self, candidate_length: int, context_length: int
    ) -> float:
        """Calculate coherence penalty based on length mismatch."""
        length_diff = abs(candidate_length - context_length)
        window = self.config.coherence_window_size

        if length_diff > window:
            return self.config.coherence_penalty_factor * (length_diff - window)
        return 0.0

    def calculate_repetition_penalty(
        self, candidate_text: str, context: Dict[str, Any]
    ) -> float:
        """Calculate repetition penalty using n-gram analysis."""
        words = candidate_text.lower().split()
        if not words:
            return 0.0

        penalty = 0.0
        history = context.get("generation_history", [])

        for n in self.config.ngram_sizes:
            if len(words) < n:
                continue

            # Extract n-grams from candidate
            candidate_ngrams = set(
                tuple(words[i : i + n]) for i in range(len(words) - n + 1)
            )

            # Check against history
            for hist_text in history[-self.config.repetition_window_size :]:
                hist_words = hist_text.lower().split()
                hist_ngrams = set(
                    tuple(hist_words[i : i + n]) for i in range(len(hist_words) - n + 1)
                )

                overlap = len(candidate_ngrams & hist_ngrams)
                if overlap > 0:
                    penalty += (
                        self.config.repetition_penalty_factor
                        * overlap
                        / len(candidate_ngrams)
                    )

        return penalty

    def calculate_diversity_penalty(
        self,
        candidate_embedding: torch.Tensor,
        previous_embeddings: torch.Tensor,
        threshold: Optional[float] = None,
    ) -> float:
        """Calculate diversity penalty using semantic similarity."""
        if previous_embeddings is None or len(previous_embeddings) == 0:
            return 0.0

        threshold = threshold or self.config.diversity_threshold

        # Calculate similarities
        similarities = SimilarityCalculator.cosine_similarity_batch(
            candidate_embedding, previous_embeddings
        )

        max_sim = similarities.max().item()

        if max_sim > threshold:
            return self.config.diversity_penalty_factor * (max_sim - threshold)
        return 0.0

    def calculate_oov_penalty(
        self, candidate_tokens: List[int], vocab: Set[int]
    ) -> float:
        """Calculate out-of-vocabulary penalty."""
        if not candidate_tokens or not vocab:
            return 0.0

        oov_count = sum(1 for t in candidate_tokens if t not in vocab)

        if self.config.oov_penalty_curve == "quadratic":
            return self.config.oov_penalty_value * (oov_count**2)
        else:  # linear
            return self.config.oov_penalty_value * oov_count

    def calculate_risk_penalty(self, candidate_text: str) -> float:
        """Calculate risk penalty based on blacklist and safety checks."""
        words = set(candidate_text.lower().split())
        blacklist_hits = words & self.config.risk_blacklist

        if blacklist_hits:
            return self.config.risk_penalty_value

        # Additional safety checks could be added here
        if self.config.enable_advanced_safety:
            # Placeholder for advanced safety model
            pass

        return 0.0

    def calculate_fluency_penalty(self, candidate_text: str) -> float:
        """Calculate fluency penalty based on language model perplexity (simplified)."""
        # This is a simplified version - in practice, you'd use a language model
        words = candidate_text.split()
        if not words:
            return 0.0

        # Simple heuristic: penalize very short words or excessive punctuation
        avg_word_len = np.mean([len(w) for w in words])
        if avg_word_len < 3:
            return self.config.fluency_penalty_factor * (3 - avg_word_len)

        return 0.0


# ==================== MAIN SCORER IMPLEMENTATION ====================


class VulcanCandidateScorer:
    """
    Advanced parallel candidate scorer with comprehensive features.

    This is the main scoring engine that orchestrates all components:
    - Embedding generation with multiple architectures
    - Similarity calculations
    - Penalty computations
    - Caching and optimization
    - Parallel execution
    """

    def __init__(self, config: Optional[VulcanScorerConfig] = None):
        """Initialize the Vulcan scorer."""
        self.config = config or VulcanScorerConfig()

        # Setup device
        self.device = self.config.device.get_device()
        log.info(f"Vulcan Scorer initialized on device: {self.device}")

        # Initialize tokenizer
        self.tokenizer = SimpleTokenizer(self.config.embedding.vocab_size)

        # Initialize embedder
        self.embedder = self._create_embedder()

        # Initialize cache
        self.cache = EmbeddingCache(self.config.cache)

        # Initialize penalty calculator
        self.penalty_calc = PenaltyCalculator(self.config.penalty)

        # Initialize feature extractor
        self.feature_extractor = StatisticalFeatureExtractor()

        # Performance tracking
        self.metrics = {
            "total_scores": 0,
            "total_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        # Thread pool for parallel execution
        if self.config.performance.use_process_pool:
            self.executor = ProcessPoolExecutor(
                max_workers=self.config.performance.max_workers
            )
        else:
            self.executor = ThreadPoolExecutor(
                max_workers=self.config.performance.max_workers
            )

        log.info("Vulcan Scorer initialization complete")

    def _create_embedder(self) -> BasedEmbedder:
        """Create embedder based on configuration."""
        embedder = EmbedderFactory.create(self.config.embedding, self.device)

        # Load pretrained weights if specified
        if (
            self.config.embedding.use_pretrained
            and self.config.embedding.pretrained_model_path
        ):
            try:
                embedder.load_state_dict(
                    torch.load(
                        self.config.embedding.pretrained_model_path,
                        map_location=self.device,
                        weights_only=True,
                    )
                )
                log.info(
                    f"Loaded pretrained weights from {self.config.embedding.pretrained_model_path}"
                )
            except Exception as e:
                log.warning(f"Failed to load pretrained weights: {e}")

        embedder.eval()
        return embedder

    def _get_embedding(
        self, text: Union[str, List[int]], use_cache: bool = True
    ) -> torch.Tensor:
        """Get embedding with caching."""
        if use_cache:
            cached = self.cache.get(text)
            if cached is not None:
                self.metrics["cache_hits"] += 1
                return cached
            self.metrics["cache_misses"] += 1

        # Generate embedding
        embedding = self.embedder.encode(text)

        if use_cache:
            self.cache.put(text, embedding)

        return embedding

    def _normalize_candidate(self, candidate: CandidateType) -> Tuple[str, List[int]]:
        """Normalize candidate to text and tokens."""
        if isinstance(candidate, str):
            text = candidate
            tokens = self.tokenizer.encode(text)
        elif isinstance(candidate, int):
            text = str(candidate)
            tokens = [candidate]
        elif isinstance(candidate, list):
            if all(isinstance(t, int) for t in candidate):
                tokens = candidate
                text = self.tokenizer.decode(tokens)
            else:
                text = " ".join(map(str, candidate))
                tokens = self.tokenizer.encode(text)
        else:
            text = str(candidate)
            tokens = []

        return text, tokens

    def _calculate_semantic_score(
        self,
        candidate_embedding: torch.Tensor,
        prompt_embedding: torch.Tensor,
        context: Dict[str, Any],
    ) -> float:
        """Calculate semantic similarity score."""
        return SimilarityCalculator.cosine_similarity(
            candidate_embedding, prompt_embedding
        )

    def _calculate_statistical_score(
        self, candidate_text: str, context: Dict[str, Any]
    ) -> float:
        """Calculate statistical quality score."""
        features = self.feature_extractor.extract_features(candidate_text, context)

        # Weighted combination of statistical features
        score = 0.0
        score += 0.3 * min(features["unique_word_ratio"], 1.0)
        score += 0.2 * min(features["entropy"] / 5.0, 1.0)
        score += 0.3 * (1.0 - min(features["punctuation_ratio"] * 5, 1.0))
        score += 0.2 * min(features["avg_word_length"] / 10.0, 1.0)

        return score

    def _calculate_all_penalties(
        self,
        candidate_text: str,
        candidate_tokens: List[int],
        candidate_embedding: torch.Tensor,
        context: Dict[str, Any],
    ) -> Dict[str, float]:
        """Calculate all penalties for a candidate."""
        penalties = {}

        candidate_length = len(candidate_text.split())
        prompt_length = len(context.get("prompt_text", "").split())

        # Length penalty
        penalties["length"] = self.penalty_calc.calculate_length_penalty(
            candidate_length, prompt_length
        )

        # Coherence penalty
        penalties["coherence"] = self.penalty_calc.calculate_coherence_penalty(
            candidate_length, prompt_length
        )

        # Diversity penalty
        previous_candidates = context.get("previous_candidates", [])
        if previous_candidates:
            prev_embeddings = []
            for prev in previous_candidates[
                -self.config.penalty.diversity_window_size :
            ]:
                prev_text, _ = self._normalize_candidate(prev)
                prev_emb = self._get_embedding(prev_text)
                prev_embeddings.append(prev_emb)

            if prev_embeddings:
                prev_embeddings_tensor = torch.cat(prev_embeddings, dim=0)
                penalties["diversity"] = self.penalty_calc.calculate_diversity_penalty(
                    candidate_embedding, prev_embeddings_tensor
                )
            else:
                penalties["diversity"] = 0.0
        else:
            penalties["diversity"] = 0.0

        # OOV penalty
        vocab = context.get("vocab", set(range(self.config.embedding.vocab_size)))
        penalties["oov"] = self.penalty_calc.calculate_oov_penalty(
            candidate_tokens, vocab
        )

        # Risk penalty
        penalties["risk"] = self.penalty_calc.calculate_risk_penalty(candidate_text)

        # Repetition penalty
        penalties["repetition"] = self.penalty_calc.calculate_repetition_penalty(
            candidate_text, context
        )

        # Fluency penalty
        penalties["fluency"] = self.penalty_calc.calculate_fluency_penalty(
            candidate_text
        )

        return penalties

    def _aggregate_scores(
        self,
        semantic_score: float,
        statistical_score: float,
        penalties: Dict[str, float],
    ) -> Tuple[float, Dict[str, Any]]:
        """Aggregate scores and penalties into final score."""
        # Calculate weighted base score
        base_score = (
            semantic_score * self.config.scoring.semantic_weight
            + statistical_score * self.config.scoring.statistical_weight
        )

        # Sum all penalties
        total_penalty = sum(penalties.values())

        # Calculate final score
        final_score = base_score - total_penalty

        # Normalize if enabled
        if self.config.scoring.enable_normalization:
            if self.config.scoring.normalization_method == "sigmoid":
                final_score = 1 / (1 + math.exp(-final_score))
            elif self.config.scoring.normalization_method == "tanh":
                final_score = math.tanh(final_score)

        # Clip to range
        min_score, max_score = self.config.scoring.score_clip_range
        final_score = max(min_score, min(max_score, final_score))

        # Build metadata
        metadata = {
            "semantic_score": semantic_score,
            "statistical_score": statistical_score,
            "base_score": base_score,
            "total_penalty": total_penalty,
            "final_score": final_score,
            "penalties": penalties,
        }

        return final_score, metadata

    def score_candidate(
        self, hidden_state: Any, candidate: CandidateType, context: Dict[str, Any]
    ) -> ScoreResult:
        """
        Score a single candidate.

        Args:
            hidden_state: Model hidden state (unused in current implementation)
            candidate: Candidate to score (text, tokens, or mixed)
            context: Context dictionary with prompt and history

        Returns:
            Tuple of (score, metadata)
        """
        start_time = time.time()

        try:
            # Normalize candidate
            candidate_text, candidate_tokens = self._normalize_candidate(candidate)

            # Get prompt embedding
            prompt_text = context.get("prompt_text", "")
            prompt_tokens = context.get("prompt_tokens", [])

            if not prompt_text and not prompt_tokens:
                log.warning("No prompt provided in context")
                return -1.0, {"error": "No prompt in context"}

            prompt_input = prompt_text if prompt_text else prompt_tokens
            prompt_embedding = self._get_embedding(prompt_input)

            # Get candidate embedding
            candidate_embedding = self._get_embedding(candidate_text)

            # Calculate semantic score
            semantic_score = self._calculate_semantic_score(
                candidate_embedding, prompt_embedding, context
            )

            # Calculate statistical score
            statistical_score = 0.0
            if self.config.scoring.strategy in [
                ScoringStrategy.STATISTICAL,
                ScoringStrategy.HYBRID,
            ]:
                statistical_score = self._calculate_statistical_score(
                    candidate_text, context
                )

            # Calculate penalties
            penalties = self._calculate_all_penalties(
                candidate_text, candidate_tokens, candidate_embedding, context
            )

            # Aggregate scores
            final_score, metadata = self._aggregate_scores(
                semantic_score, statistical_score, penalties
            )

            # Update metrics
            elapsed = time.time() - start_time
            self.metrics["total_scores"] += 1
            self.metrics["total_time"] += elapsed

            if self.config.verbose:
                log.debug(f"Scored candidate in {elapsed:.4f}s: {final_score:.4f}")

            return final_score, metadata

        except Exception as e:
            log.error(f"Error scoring candidate: {e}", exc_info=True)
            return -0.5, {"error": str(e)}

    async def score_candidates_async(
        self,
        hidden_state: Any,
        candidates: Sequence[CandidateType],
        context: Dict[str, Any],
    ) -> List[float]:
        """
        Score multiple candidates in parallel asynchronously.

        Args:
            hidden_state: Model hidden state
            candidates: List of candidates to score
            context: Context dictionary

        Returns:
            List of scores
        """
        if not candidates:
            return []

        loop = asyncio.get_running_loop()
        timeout = self.config.performance.parallel_timeout_seconds

        # Create scoring tasks
        tasks = []
        for candidate in candidates:
            task = loop.run_in_executor(
                self.executor, self.score_candidate, hidden_state, candidate, context
            )
            tasks.append(task)

        # Execute tasks with timeout
        results = []
        for task in tasks:
            try:
                result = await asyncio.wait_for(task, timeout=timeout)
                results.append(result)
            except asyncio.TimeoutError:
                log.warning(f"Candidate scoring timed out after {timeout}s")
                results.append((-0.5, {"error": "timeout"}))
            except Exception as e:
                log.error(f"Error in async scoring: {e}")
                results.append((-0.5, {"error": str(e)}))

        # Extract scores
        scores = [r[0] if isinstance(r, tuple) else r for r in results]

        return scores

    def score_candidates_batch(
        self,
        hidden_state: Any,
        candidates: Sequence[CandidateType],
        context: Dict[str, Any],
        batch_size: Optional[int] = None,
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Score candidates in batches for efficiency.

        Args:
            hidden_state: Model hidden state
            candidates: List of candidates to score
            context: Context dictionary
            batch_size: Batch size (defaults to config value)

        Returns:
            List of (score, metadata) tuples
        """
        if not candidates:
            return []

        batch_size = batch_size or self.config.performance.batch_size
        results = []

        for i in range(0, len(candidates), batch_size):
            batch = candidates[i : i + batch_size]
            batch_results = [
                self.score_candidate(hidden_state, candidate, context)
                for candidate in batch
            ]
            results.extend(batch_results)

        return results

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        cache_stats = self.cache.get_stats()

        avg_time = (
            self.metrics["total_time"] / self.metrics["total_scores"]
            if self.metrics["total_scores"] > 0
            else 0.0
        )

        return {
            "total_scores": self.metrics["total_scores"],
            "total_time": self.metrics["total_time"],
            "avg_time_per_score": avg_time,
            "cache_stats": cache_stats,
            "scores_per_second": (
                self.metrics["total_scores"] / self.metrics["total_time"]
                if self.metrics["total_time"] > 0
                else 0.0
            ),
        }

    def reset_metrics(self):
        """Reset performance metrics."""
        self.metrics = {
            "total_scores": 0,
            "total_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
        }
        log.info("Metrics reset")

    def clear_cache(self):
        """Clear the embedding cache."""
        self.cache.clear()

    def save_state(self, path: str):
        """Save scorer state to disk."""
        state = {
            "config": asdict(self.config),
            "metrics": self.metrics,
            "tokenizer_vocab": self.tokenizer.word_to_id,
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(state, f)

        # Save model weights
        model_path = Path(path).with_suffix(".pt")
        torch.save(self.embedder.state_dict(), model_path)

        log.info(f"Scorer state saved to {path}")

    def load_state(self, path: str):
        """Load scorer state from disk."""
        with open(path, "rb") as f:
            state = safe_pickle_load(f)

        self.metrics = state["metrics"]
        self.tokenizer.word_to_id = state["tokenizer_vocab"]
        self.tokenizer.id_to_word = {v: k for k, v in self.tokenizer.word_to_id.items()}

        # Load model weights
        model_path = Path(path).with_suffix(".pt")
        if model_path.exists():
            self.embedder.load_state_dict(
                torch.load(model_path, map_location=self.device, weights_only=True)
            )

        log.info(f"Scorer state loaded from {path}")

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False)


# ==================== CONVENIENCE FUNCTIONS ====================

# Global scorer instance (lazy initialization)
_global_scorer: Optional[VulcanCandidateScorer] = None
_scorer_lock = threading.Lock()


def get_global_scorer(
    config: Optional[VulcanScorerConfig] = None,
) -> VulcanCandidateScorer:
    """Get or create the global scorer instance."""
    global _global_scorer

    if _global_scorer is None:
        with _scorer_lock:
            if _global_scorer is None:
                _global_scorer = VulcanCandidateScorer(config)

    return _global_scorer


async def score_candidates(
    hidden_state: Any,
    candidates: Sequence[CandidateType],
    context: Dict[str, Any],
    config: Optional[VulcanScorerConfig] = None,
) -> List[float]:
    """
    Convenience function for scoring candidates.

    This function maintains backward compatibility with the original API.

    Args:
        hidden_state: Model hidden state
        candidates: List of candidates to score
        context: Context dictionary with prompt and history
        config: Optional configuration (uses global config if None)

    Returns:
        List of scores
    """
    scorer = get_global_scorer(config)
    return await scorer.score_candidates_async(hidden_state, candidates, context)


def score_candidate_sync(
    hidden_state: Any,
    candidate: CandidateType,
    context: Dict[str, Any],
    config: Optional[VulcanScorerConfig] = None,
) -> ScoreResult:
    """
    Synchronous convenience function for scoring a single candidate.

    Args:
        hidden_state: Model hidden state
        candidate: Candidate to score
        context: Context dictionary
        config: Optional configuration

    Returns:
        Tuple of (score, metadata)
    """
    scorer = get_global_scorer(config)
    return scorer.score_candidate(hidden_state, candidate, context)


# ==================== BACKWARDS COMPATIBILITY ====================

# Maintain compatibility with original function names
semantic_candidate_scorer = score_candidate_sync

# Export original config for compatibility
CandidateScoreConfig = VulcanScorerConfig

# ==================== MAIN ENTRY POINT ====================


def main():
    """Main entry point for testing and demonstration."""
    import argparse

    parser = argparse.ArgumentParser(description="Vulcan Candidate Scorer")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarks")
    args = parser.parse_args()

    # Load configuration
    if args.config:
        config = VulcanScorerConfig.load(args.config)
    else:
        config = VulcanScorerConfig()

    # Create scorer
    scorer = VulcanCandidateScorer(config)

    if args.test:
        print("Running tests...")

        # Test context
        context = {
            "prompt_text": "The quick brown fox jumps over the lazy dog",
            "prompt_tokens": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "previous_candidates": [],
            "vocab": set(range(10000)),
        }

        # Test candidates
        candidates = [
            "A fast fox leaps over a sleeping dog",
            "The cat sits on the mat",
            "Hello world",
            "Machine learning is fascinating",
            [10, 20, 30, 40, 50],
        ]

        print("\nScoring candidates...")
        for candidate in candidates:
            score, metadata = scorer.score_candidate(None, candidate, context)
            print(f"\nCandidate: {candidate}")
            print(f"Score: {score:.4f}")
            print(f"Metadata: {json.dumps(metadata, indent=2, default=str)}")

        # Test async scoring
        async def test_async():
            scores = await scorer.score_candidates_async(None, candidates, context)
            print("\nAsync scores:", scores)

        asyncio.run(test_async())

        # Print metrics
        print("\nMetrics:")
        print(json.dumps(scorer.get_metrics(), indent=2))

    if args.benchmark:
        print("Running benchmarks...")

        import random

        context = {"prompt_text": "Benchmark prompt " * 10, "previous_candidates": []}

        # Generate random candidates
        num_candidates = 1000
        candidates = [
            f"Candidate {i} with some random text "
            + " ".join(f"word{random.randint(0, 1000)}" for _ in range(10))
            for i in range(num_candidates)
        ]

        print(f"\nBenchmarking {num_candidates} candidates...")
        start_time = time.time()

        results = scorer.score_candidates_batch(None, candidates, context)

        elapsed = time.time() - start_time
        print(f"Total time: {elapsed:.2f}s")
        print(f"Candidates per second: {num_candidates / elapsed:.2f}")
        print(f"Average time per candidate: {elapsed / num_candidates * 1000:.2f}ms")

        # Print metrics
        print("\nMetrics:")
        print(json.dumps(scorer.get_metrics(), indent=2))


if __name__ == "__main__":
    main()
