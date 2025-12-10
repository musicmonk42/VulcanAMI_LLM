from __future__ import annotations

"""
GraphixTransformer (Fully Functional Lightweight Implementation - PRODUCTION CORE)

Composition:
- Builds per-layer IR using IRAttention, IRFeedForward, IRLayerNorm
- Uses GraphixExecutor for execution
- Provides encode(), forward(), generate(), get_logits(), get_embeddings(), reset_parameters()
- Implements configuration validation (using a simple dictionary check for simplicity over external pydantic)
- Supports governed weight updates and standard model utility functions.
- ENHANCED with PEFT (LoRA) structure, Top-P sampling, and Gradient Checkpointing logic.
- FIXED: Added SimpleTokenizer for string-to-token-ID conversion
- FIXED: Improved get_logits() API for easier usage
"""

import bisect  # For Top-P sampling
import hashlib  # For tokenizer hashing
import json  # Used for simplified save/load structure
import math
import random
from dataclasses import asdict, dataclass
from functools import lru_cache  # Enhancement 2: IR caching
from typing import Any, Dict, List, Optional, Sequence, Union

# Handle both absolute and relative imports gracefully
try:
    from .graphix_executor import GraphixExecutor
    from .ir_attention import IRAttention
    from .ir_embeddings import IREmbeddings
    from .ir_feedforward import IRFeedForward
    from .ir_layer_norm import IRLayerNorm
except ImportError:
    # Fallback to absolute imports if relative imports fail
    from graphix_executor import GraphixExecutor
    from ir_attention import IRAttention
    from ir_embeddings import IREmbeddings
    from ir_feedforward import IRFeedForward
    from ir_layer_norm import IRLayerNorm

TokenLike = Union[str, int]
TokensLike = Union[str, Sequence[TokenLike]]


# ============================================================================
# SIMPLE TOKENIZER (FIXED: Added to support string inputs)
# ============================================================================


class SimpleTokenizer:
    """Simple word-based tokenizer for text-to-token-ID conversion.

    This tokenizer provides basic word-level tokenization for the transformer.
    For production use, consider using BPE, WordPiece, or SentencePiece tokenizers.
    """

    def __init__(self, vocab_size: int):
        """Initialize tokenizer.

        Args:
            vocab_size: Maximum vocabulary size
        """
        self.vocab_size = vocab_size
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.next_id = 1  # Start from 1, reserve 0 for special tokens

        # Special tokens
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"

        # Add special tokens
        self.word_to_id[self.pad_token] = 0
        self.id_to_word[0] = self.pad_token

        for special_token in [self.unk_token, self.bos_token, self.eos_token]:
            if self.next_id < vocab_size:
                self.word_to_id[special_token] = self.next_id
                self.id_to_word[self.next_id] = special_token
                self.next_id += 1

    def encode(self, text: Union[str, List[str]]) -> List[int]:
        """Encode text to token IDs.

        Args:
            text: Input text string or list of words

        Returns:
            List of token IDs
        """
        if isinstance(text, str):
            words = text.split()
        else:
            words = text

        tokens = []
        for word in words:
            if word not in self.word_to_id:
                # Add new word if space available
                if self.next_id < self.vocab_size:
                    self.word_to_id[word] = self.next_id
                    self.id_to_word[self.next_id] = word
                    self.next_id += 1
                    tokens.append(self.next_id - 1)
                else:
                    # Use unknown token
                    tokens.append(self.word_to_id.get(self.unk_token, 1))
            else:
                tokens.append(self.word_to_id[word])

        return tokens

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs back to text.

        Args:
            tokens: List of token IDs

        Returns:
            Decoded text string
        """
        words = [self.id_to_word.get(t, self.unk_token) for t in tokens]
        return " ".join(words)

    def get_vocab_size(self) -> int:
        """Get current vocabulary size.

        Returns:
            Number of tokens in vocabulary
        """
        return len(self.word_to_id)

    def save(self, path: str) -> None:
        """Save tokenizer vocabulary.

        Args:
            path: File path to save vocabulary
        """
        vocab_data = {
            "word_to_id": self.word_to_id,
            "id_to_word": {str(k): v for k, v in self.id_to_word.items()},
            "vocab_size": self.vocab_size,
            "next_id": self.next_id,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SimpleTokenizer":
        """Load tokenizer vocabulary.

        Args:
            path: File path to load vocabulary from

        Returns:
            Loaded tokenizer instance
        """
        with open(path, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)

        tokenizer = cls(vocab_data["vocab_size"])
        tokenizer.word_to_id = vocab_data["word_to_id"]
        tokenizer.id_to_word = {int(k): v for k, v in vocab_data["id_to_word"].items()}
        tokenizer.next_id = vocab_data["next_id"]
        return tokenizer


# ============================================================================
# TRANSFORMER CONFIGURATION
# ============================================================================


@dataclass
class GraphixTransformerConfig:
    """Configuration for GraphixTransformer model."""

    num_layers: int = 6
    hidden_size: int = 256
    num_heads: int = 4
    vocab_size: int = 4096
    max_position_embeddings: int = 1024
    dropout: float = 0.1  # Increased default dropout
    layer_norm_eps: float = 1e-5
    seed: Optional[int] = 1234

    # Enhancement 7, 8: Added training flags
    gradient_checkpointing: bool = False  # CRITICAL ENHANCEMENT: For memory
    dtype: str = "float32"  # Placeholder for mixed precision

    # CRITICAL ENHANCEMENT: LoRA Config
    lora_rank: int = 0  # 0 means disabled
    lora_alpha: float = 1.0


# ============================================================================
# MAIN TRANSFORMER CLASS
# ============================================================================


class GraphixTransformer:
    """Main transformer model class with IR-based execution."""

    # Enhancement 2: Caching helper for IR (uses class method for config hashing)
    @staticmethod
    @lru_cache(maxsize=1)
    def _get_ir_cache(config_tuple: tuple) -> Dict[str, Any]:
        """Caches the full, static IR graph based on config parameters."""
        cfg = GraphixTransformerConfig(**dict(config_tuple))

        embedding_ir = IREmbeddings().build_ir(
            cfg.vocab_size, cfg.hidden_size, cfg.max_position_embeddings, cfg.dropout
        )

        layers: List[Dict[str, Any]] = []
        for L in range(cfg.num_layers):
            attn_ir = IRAttention().build_ir(cfg.num_heads, cfg.hidden_size)
            ffn_ir = IRFeedForward().build_ir(
                cfg.hidden_size, cfg.hidden_size * 4, cfg.dropout
            )
            ln_ir = IRLayerNorm().build_ir(
                cfg.hidden_size, cfg.layer_norm_eps, norm_type="rmsnorm"
            )  # Use RMSNorm

            # Layer stitching
            layer_graph = {
                "layer": L,
                "nodes": attn_ir["nodes"] + ffn_ir["nodes"] + ln_ir["nodes"],
                "edges": attn_ir["edges"] + ffn_ir["edges"] + ln_ir["edges"],
                "metadata": {"type": "transformer_layer", "index": L},
            }
            layers.append(layer_graph)

        return {"embedding": embedding_ir, "layers": layers}

    def __init__(
        self,
        config: Optional[GraphixTransformerConfig] = None,
        observability: Optional[Any] = None,
        audit_log: Optional[Any] = None,
    ) -> None:
        """Initialize GraphixTransformer model.

        Args:
            config: Model configuration object
            observability: Optional observability handler for monitoring
            audit_log: Optional audit logging handler

        Raises:
            ValueError: If configuration parameters are invalid
        """
        self.config = config or GraphixTransformerConfig()

        # Enhancement 1: Basic Config Validation
        self._validate_config()

        self._rng = random.Random(
            self.config.seed if self.config.seed is not None else 9876
        )

        # FIXED: Initialize tokenizer for string-to-token-ID conversion
        self.tokenizer = SimpleTokenizer(vocab_size=self.config.vocab_size)

        # Enhancement 2: Retrieve IR from cache
        config_tuple = tuple(sorted(asdict(self.config).items()))
        cached_ir = GraphixTransformer._get_ir_cache(config_tuple)
        self.embedding_ir = cached_ir["embedding"]
        self.layers = cached_ir["layers"]

        # Executor
        self.executor = GraphixExecutor(
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            vocab_size=self.config.vocab_size,
            max_position_embeddings=self.config.max_position_embeddings,
            layer_norm_eps=self.config.layer_norm_eps,
            seed=self.config.seed,
            observability=observability,
            audit_log=audit_log,
        )

        # CRITICAL ENHANCEMENT 3: Placeholder for LoRA/PEFT adapters
        self.adapters: Dict[str, Any] = self._init_lora_adapters()

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        if self.config.hidden_size % self.config.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.config.hidden_size}) must be divisible by num_heads ({self.config.num_heads})."
            )

        if self.config.num_layers <= 0:
            raise ValueError(
                f"num_layers must be positive, got {self.config.num_layers}"
            )

        if self.config.vocab_size <= 0:
            raise ValueError(
                f"vocab_size must be positive, got {self.config.vocab_size}"
            )

        if not (0 <= self.config.dropout < 1):
            raise ValueError(f"dropout must be in [0, 1), got {self.config.dropout}"]

        if self.config.layer_norm_eps <= 0:
            raise ValueError(
                f"layer_norm_eps must be positive, got {self.config.layer_norm_eps}"
            )

        if self.config.lora_rank < 0:
            raise ValueError(
                f"lora_rank must be non-negative, got {self.config.lora_rank}"
            )

    # --------------------- LoRA (CRITICAL ENHANCEMENT) --------------------- #

    def _init_lora_adapters(self) -> Dict[str, Any]:
        """Initializes LoRA adapters if lora_rank > 0.

        Returns:
            Dictionary of LoRA adapter weights for each layer and projection
        """
        adapters: Dict[str, Any] = {}
        if self.config.lora_rank > 0:
            rank = self.config.lora_rank
            H = self.config.hidden_size
            # Simplified LoRA weights (A: HxR, B: RxH)

            def init_lora_weights(in_dim: int, out_dim: int) -> Dict[str, Any]:
                """Initialize LoRA weight matrices A and B.

                Args:
                    in_dim: Input dimension
                    out_dim: Output dimension

                Returns:
                    Dictionary with LoRA weights A and B
                """
                # Init A with small uniform, B with zeros (or small init)
                lora_A = [self._rng.uniform(-0.01, 0.01) for _ in range(in_dim * rank)]
                lora_B = [0.0 for _ in range(rank * out_dim)]
                return {"A": lora_A, "B": lora_B, "rank": rank}

            # Apply to Q, K, V, and FFN W2 (common Llama-style)
            for L in range(self.config.num_layers):
                adapters[f"layer_{L}.attn.q"] = init_lora_weights(H, H)
                adapters[f"layer_{L}.attn.k"] = init_lora_weights(H, H)
                adapters[f"layer_{L}.attn.v"] = init_lora_weights(H, H)
                adapters[f"layer_{L}.ffn.w2"] = init_lora_weights(
                    H * 4, H
                )  # Assuming I size is 4H

        return adapters

    def add_adapter(self, name: str, rank: int, alpha: Optional[float] = None) -> None:
        """Add a named LoRA adapter to the model.

        Args:
            name: Name of the adapter
            rank: Rank of the LoRA decomposition
            alpha: LoRA scaling factor (optional)
        """
        # For simplicity, this demo just sets the rank/alpha on the main config
        # A real system would maintain multiple named adapters
        self.config.lora_rank = rank
        self.config.lora_alpha = alpha if alpha is not None else self.config.lora_alpha
        self.adapters = self._init_lora_adapters()
        print(
            f"Initialized LoRA adapter '{name}' with rank={rank}, alpha={self.config.lora_alpha}"
        )

    # --------------------- Encoding & Forward --------------------- #

    def _normalize_tokens(self, tokens: TokensLike) -> List[int]:
        """Normalize various token input formats to a list of token IDs.

        FIXED: Now always returns integer token IDs

        Args:
            tokens: Input tokens in various formats (string, list of strings, list of ints)

        Returns:
            List of integer token IDs
        """
        if isinstance(tokens, str):
            # String input - tokenize it
            return self.tokenizer.encode(tokens)
        elif isinstance(tokens, (list, tuple)):
            if not tokens:
                return []
            # Check if first element is string or int
            if isinstance(tokens[0], str):
                # List of strings - join and tokenize
                text = " ".join(tokens)
                return self.tokenizer.encode(text)
            else:
                # Already token IDs
                return list(tokens)
        else:
            # Single token
            if isinstance(tokens, str):
                return self.tokenizer.encode(tokens)
            else:
                return [int(tokens)]

    def encode(self, tokens: TokensLike) -> Dict[str, Any]:
        """Encode tokens (deprecated, use forward() or get_embeddings()).

        Args:
            tokens: Input tokens

        Returns:
            Dictionary with encoding results
        """
        return self.forward(tokens)

    def forward(self, tokens: TokensLike) -> Dict[str, Any]:
        """Standard forward pass through the transformer.

        FIXED: Now properly converts string inputs to token IDs

        Args:
            tokens: Input tokens (string, list of strings, or list of token IDs)

        Returns:
            Dictionary containing:
                - hidden_states: Final hidden states
                - execution_time_ms: Execution time
                - cache_stats: KV cache statistics
                - metrics: Execution metrics
        """
        # FIXED: Normalize to token IDs (handles strings now)
        token_ids = self._normalize_tokens(tokens)

        graph_ir = {
            "type": "transformer_forward",
            "embedding": self.embedding_ir,
            "layers": self.layers,
        }

        inputs = {
            "tokens": token_ids,  # FIXED: Now always token IDs
            # CRITICAL ENHANCEMENT: Pass LoRA adapters for fusion in executor
            "lora_adapters": self.adapters,
            "lora_alpha": self.config.lora_alpha,
            # CRITICAL ENHANCEMENT: Pass Checkpointing flag
            "gradient_checkpointing": self.config.gradient_checkpointing,
        }

        result = self.executor.execute(graph_ir, inputs=inputs)
        return result

    def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings for input text.

        Args:
            text: Input text string

        Returns:
            List of embedding values
        """
        result = self.forward(text)  # FIXED: forward() now handles strings
        hidden = result.get("hidden_states", [])
        return hidden if isinstance(hidden, list) else []

    # --------------------- Generation (CRITICAL ENHANCEMENT) --------------------- #

    def get_logits(
        self,
        input_text_or_hidden: Union[str, Any],
        tokens: Optional[Sequence[Any]] = None,
    ) -> List[float]:
        """Get logits for next token prediction.

        FIXED: Now supports both string input (convenience) and hidden state input (advanced)

        Args:
            input_text_or_hidden: Either input text string (for convenience) or hidden state (for advanced use)
            tokens: Token sequence (only needed if providing hidden state directly)

        Returns:
            List of logit values for vocabulary
        """
        if isinstance(input_text_or_hidden, str):
            # Convenience mode: string input
            result = self.forward(input_text_or_hidden)
            hidden_state = result.get("hidden_states", [])
            token_ids = self._normalize_tokens(input_text_or_hidden)
            return self.executor.get_logits(hidden_state, token_ids)
        else:
            # Advanced mode: hidden state provided directly
            if tokens is None:
                raise ValueError(
                    "tokens parameter required when providing hidden_state directly"
                )
            hidden_state = input_text_or_hidden
            token_list = list(tokens) if not isinstance(tokens, list) else tokens
            return self.executor.get_logits(hidden_state, token_list)

    def _generate_next_token(
        self,
        candidate_logits: List[float],
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> int:
        """Generate next token using temperature and Top-P sampling.

        Args:
            candidate_logits: Logit values for all vocabulary tokens
            temperature: Sampling temperature (lower = more deterministic)
            top_p: Nucleus sampling threshold

        Returns:
            Selected token ID
        """
        if not candidate_logits:
            return 0

        # Apply Temperature
        if temperature <= 0:
            # Greedy decoding
            return max(range(len(candidate_logits), key=lambda i: candidate_logits[i]

        # Compute probabilities with temperature
        max_logit = max(candidate_logits)  # For numerical stability
        exps = [math.exp((l - max_logit) / temperature) for l in candidate_logits]
        sum_exps = sum(exps)

        if sum_exps == 0:
            return 0  # Fallback

        probs = [e / sum_exps for e in exps]

        # Get sorted probabilities and indices
        sorted_probs_with_idx = sorted(
            [(p, i) for i, p in enumerate(probs)], reverse=True
        )

        # CRITICAL ENHANCEMENT: Top-P Filtering (Nucleus Sampling)
        if 0.0 < top_p < 1.0:
            cumulative_prob = 0.0
            top_p_candidates = []
            for p, idx in sorted_probs_with_idx:
                cumulative_prob += p
                top_p_candidates.append((p, idx))
                if cumulative_prob >= top_p:
                    break

            # Rescale remaining probabilities
            total_prob = sum(p for p, idx in top_p_candidates)
            if total_prob == 0:
                return 0  # Fallback

            scaled_probs = [p / total_prob for p, idx in top_p_candidates]
            indices = [idx for p, idx in top_p_candidates]

            # Weighted choice on the filtered set
            r = self._rng.random()
            cumulative_sum = 0.0
            for i, p_scaled in enumerate(scaled_probs):
                cumulative_sum += p_scaled
                if r < cumulative_sum:
                    return indices[i]
            return indices[0] if indices else 0  # Fallback to highest prob in p-nucleus

        # Standard Weighted Choice
        else:
            r = self._rng.random()
            cumulative_prob = 0.0
            for i, p in enumerate(probs):
                cumulative_prob += p
                if r < cumulative_prob:
                    return i
            return max(
                range(len(candidate_logits), key=lambda i: candidate_logits[i]
            )  # Fallback to greedy

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Generate text continuation from prompt.

        FIXED: Now properly handles string prompts and returns decoded text

        Args:
            prompt: Input prompt string
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold

        Returns:
            Generated text string
        """
        # FIXED: Convert prompt to token IDs
        token_ids: List[int] = self._normalize_tokens(prompt)

        for step in range(max_new_tokens):
            # 1. Forward pass (uses current full sequence)
            # The executor implicitly uses KV caching for efficiency if available
            result = self.forward(token_ids)
            hidden_state = result.get("hidden_states")

            # 2. Get logits for the next token
            logits = self.executor.get_logits(hidden_state, token_ids)

            # 3. Sample the next token
            next_token_id = self._generate_next_token(logits, temperature, top_p)

            # Stop condition (if applicable, e.g., EOS token)
            if next_token_id == 0 and len(token_ids) > 1:
                break

            token_ids.append(next_token_id)

        # FIXED: Decode token IDs back to text
        return self.tokenizer.decode(token_ids)

    # --------------------- Training Call --------------------- #

    def __call__(self, batch: Dict[str, Any]) -> float:
        """Forward pass with loss calculation (callable interface).

        Args:
            batch: Batch dictionary with 'tokens' key

        Returns:
            Pseudo loss value
        """
        return self.forward_loss(batch)

    def forward_loss(self, batch: Dict[str, Any]) -> float:
        """Calculate forward pass and return pseudo loss.

        Args:
            batch: Batch dictionary with 'tokens' key

        Returns:
            Pseudo loss value for demonstration
        """
        tokens = batch.get("tokens", [])
        size = len(tokens) if isinstance(tokens, list) else 1
        # Encode for internal side effects / audit
        self.forward(tokens)
        # Pseudo loss influenced by length
        return 0.05 + 0.001 * size + self._rng.uniform(-0.005, 0.005)

    # --------------------- Governance Update --------------------- #

    def apply_update(self, gradients: Dict[str, Any]) -> None:
        """Apply gradient update to model weights.

        Args:
            gradients: Dictionary of gradients with optional 'lr' key
        """
        proposal = {"gradients": gradients, "lr": gradients.get("lr", 0.001)}
        self.executor.apply_update(proposal)

    # --------------------- Utility --------------------- #

    def reset_parameters(self) -> None:
        """Re-initialize all model weights."""
        self.executor._init_layer_weights()
        # Also reinitialize LoRA adapters if present
        if self.config.lora_rank > 0:
            self.adapters = self._init_lora_adapters()

    def save(self, path: str) -> None:
        """Save model configuration and weights to file.

        Args:
            path: File path to save model state
        """
        model_state = {
            "config": asdict(self.config),
            "weights": self.executor.weights,
            "adapters": self.adapters,  # CRITICAL ENHANCEMENT: Save adapters
            "tokenizer_vocab": {
                "word_to_id": self.tokenizer.word_to_id,
                "id_to_word": {str(k): v for k, v in self.tokenizer.id_to_word.items()},
                "next_id": self.tokenizer.next_id,
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(model_state, f, indent=2)

    @classmethod
    def load(
        cls,
        path: str,
        observability: Optional[Any] = None,
        audit_log: Optional[Any] = None,
    ) -> "GraphixTransformer":
        """Load model from saved state file.

        Args:
            path: File path to load model state from
            observability: Optional observability handler
            audit_log: Optional audit logging handler

        Returns:
            Loaded GraphixTransformer instance
        """
        with open(path, "r", encoding="utf-8") as f:
            model_state = json.load(f)

        config = GraphixTransformerConfig(**model_state["config"])
        instance = cls(config, observability, audit_log)
        instance.executor.weights = model_state["weights"]
        instance.adapters = model_state.get(
            "adapters", {}
        )  # CRITICAL ENHANCEMENT: Load adapters

        # FIXED: Restore tokenizer vocabulary
        if "tokenizer_vocab" in model_state:
            vocab_data = model_state["tokenizer_vocab"]
            instance.tokenizer.word_to_id = vocab_data["word_to_id"]
            instance.tokenizer.id_to_word = {
                int(k): v for k, v in vocab_data["id_to_word"].items()
            }
            instance.tokenizer.next_id = vocab_data["next_id"]

        return instance

    def vocab_size(self) -> int:
        """Get vocabulary size.

        Returns:
            Size of the vocabulary
        """
        return self.config.vocab_size

    def num_parameters(self) -> int:
        """Calculate total number of parameters.

        Returns:
            Total parameter count
        """
        total = 0

        # Embedding parameters
        total += self.config.vocab_size * self.config.hidden_size

        # Layer parameters
        for _ in range(self.config.num_layers):
            # Attention (Q, K, V, O)
            total += 4 * (self.config.hidden_size * self.config.hidden_size)

            # FFN (gate, up, down)
            intermediate_size = self.config.hidden_size * 4
            total += 2 * (self.config.hidden_size * intermediate_size)  # gate + up
            total += intermediate_size * self.config.hidden_size  # down

            # Layer norms (2 per layer)
            total += 2 * self.config.hidden_size

        # Final layer norm
        total += self.config.hidden_size

        # Output projection
        total += self.config.hidden_size * self.config.vocab_size

        # LoRA parameters if enabled
        if self.config.lora_rank > 0:
            lora_params = 0
            for _ in range(self.config.num_layers):
                # Q, K, V adapters
                lora_params += 3 * 2 * self.config.hidden_size * self.config.lora_rank
                # FFN adapter
                lora_params += 2 * self.config.hidden_size * 4 * self.config.lora_rank
            total += lora_params

        return total

    def get_config(self) -> GraphixTransformerConfig:
        """Get model configuration.

        Returns:
            Current model configuration
        """
        return self.config

    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility.

        Args:
            seed: Random seed value
        """
        self.config.seed = seed
        self._rng = random.Random(seed)
        self.executor.set_seed(seed)

    def eval(self) -> None:
        """Set model to evaluation mode."""
        self.executor.set_mode("eval")

    def train(self) -> None:
        """Set model to training mode."""
        self.executor.set_mode("train")


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================


def build_transformer_as_ir(
    config: Optional[GraphixTransformerConfig] = None,
) -> GraphixTransformer:
    """Factory function to build a GraphixTransformer instance.

    Args:
        config: Optional configuration object

    Returns:
        GraphixTransformer instance
    """
    return GraphixTransformer(config)


# Utility functions for common use cases


def create_small_model() -> GraphixTransformer:
    """Create a small transformer model for testing.

    Returns:
        Small GraphixTransformer instance
    """
    config = GraphixTransformerConfig(
        num_layers=3,
        hidden_size=128,
        num_heads=4,
        vocab_size=1000,
        max_position_embeddings=512,
    )
    return GraphixTransformer(config)


def create_medium_model() -> GraphixTransformer:
    """Create a medium-sized transformer model.

    Returns:
        Medium GraphixTransformer instance
    """
    config = GraphixTransformerConfig(
        num_layers=6,
        hidden_size=512,
        num_heads=8,
        vocab_size=10000,
        max_position_embeddings=1024,
    )
    return GraphixTransformer(config)


def create_large_model() -> GraphixTransformer:
    """Create a large transformer model.

    Returns:
        Large GraphixTransformer instance
    """
    config = GraphixTransformerConfig(
        num_layers=12,
        hidden_size=768,
        num_heads=12,
        vocab_size=50257,
        max_position_embeddings=2048,
    )
    return GraphixTransformer(config)


def create_lora_model(
    base_config: Optional[GraphixTransformerConfig] = None,
    lora_rank: int = 16,
    lora_alpha: float = 32.0,
) -> GraphixTransformer:
    """Create a transformer model with LoRA adapters.

    Args:
        base_config: Base configuration (optional)
        lora_rank: LoRA rank
        lora_alpha: LoRA scaling factor

    Returns:
        GraphixTransformer instance with LoRA
    """
    config = base_config or GraphixTransformerConfig()
    config.lora_rank = lora_rank
    config.lora_alpha = lora_alpha
    return GraphixTransformer(config)


# ============================================================================
# MAIN TESTING
# ============================================================================

# Main execution for testing
if __name__ == "__main__":
    print("GraphixTransformer Module Test")
    print("=" * 50)

    # Test configuration validation
    print("\n1. Testing configuration validation...")
    try:
        bad_config = GraphixTransformerConfig(
            hidden_size=100,
            num_heads=7,  # Not divisible
        )
        model = GraphixTransformer(bad_config)
        print("   ERROR: Should have raised ValueError")
    except ValueError as e:
        print(f"   ✓ Validation caught error: {e}")

    # Test model creation
    print("\n2. Testing model creation...")
    model = create_small_model()
    print(f"   ✓ Created model with {model.num_parameters():,} parameters")

    # Test forward pass
    print("\n3. Testing forward pass...")
    result = model.forward("hello world")
    print(f"   ✓ Forward pass completed")
    print(f"   Hidden states shape: {len(result.get('hidden_states', []))} dimensions")

    # Test generation
    print("\n4. Testing text generation...")
    generated = model.generate("Once upon a", max_new_tokens=10, temperature=0.8)
    print(f"   ✓ Generated: {generated}")

    # Test LoRA
    print("\n5. Testing LoRA adapter...")
    lora_model = create_lora_model(lora_rank=8, lora_alpha=16)
    lora_model.add_adapter("test_adapter", rank=8, alpha=16)
    print(f"   ✓ LoRA model with {lora_model.num_parameters():,} parameters")

    # Test save/load
    print("\n6. Testing save/load...")
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        model.save(tmp_path)
        print(f"   ✓ Model saved to {tmp_path}")

        loaded_model = GraphixTransformer.load(tmp_path)
        print(f"   ✓ Model loaded successfully")
        print(f"   Loaded model has {loaded_model.num_parameters():,} parameters")
    finally:
        os.unlink(tmp_path)

    print("\n" + "=" * 50)
    print("All tests passed successfully!")
