# Ultra Deep Dive Examination of Vulcan's Internal LLM

**Version:** 2.0.0  
**Last Updated:** December 30, 2024

This document provides an ultra deep dive examination into the VulcanAMI platform's internal Large Language Model (LLM) architecture, covering all components from the low-level IR representation to the high-level cognitive reasoning systems. This is an exhaustive technical reference for developers, researchers, and operators.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Core LLM Components](#3-core-llm-components)
   - [3.1 GraphixTransformer](#31-graphixtransformer)
   - [3.2 IR (Intermediate Representation) System](#32-ir-intermediate-representation-system)
   - [3.3 GraphixExecutor - Production IR Engine](#33-graphixexecutor---production-ir-engine)
   - [3.4 Local GPT Provider](#34-local-gpt-provider)
4. [Attention Mechanisms](#4-attention-mechanisms)
5. [Feed-Forward Networks](#5-feed-forward-networks)
6. [Normalization Layers](#6-normalization-layers)
7. [Embedding System](#7-embedding-system)
8. [Persistent Context Management](#8-persistent-context-management)
9. [Memory Architecture](#9-memory-architecture)
   - [9.1 Hierarchical Memory System](#91-hierarchical-memory-system)
   - [9.2 Persistent Memory with ZK Proofs](#92-persistent-memory-with-zk-proofs)
   - [9.3 Tool Selection Memory](#93-tool-selection-memory)
10. [Reasoning Systems](#10-reasoning-systems)
    - [10.1 Unified Reasoning Interface](#101-unified-reasoning-interface)
    - [10.2 Reasoning Types](#102-reasoning-types)
    - [10.3 Mathematical Verification](#103-mathematical-verification)
11. [World Model Integration](#11-world-model-integration)
    - [11.1 Core Components](#111-core-components)
    - [11.2 Meta-Reasoning](#112-meta-reasoning)
12. [Self-Improvement System](#12-self-improvement-system)
13. [Safety and Governance](#13-safety-and-governance)
14. [Performance Instrumentation](#14-performance-instrumentation)
15. [Configuration and Tuning](#15-configuration-and-tuning)
16. [Code Examples](#16-code-examples)
17. [Future Directions](#17-future-directions)

---

## 1. Executive Summary

The Vulcan Internal LLM is a sophisticated, multi-layered language model architecture that integrates:

- **Custom Transformer Architecture**: Built using an Intermediate Representation (IR) system for maximum flexibility
- **Production IR Execution Engine**: GraphixExecutor with Flash Attention, quantization, and KV caching
- **Hybrid Attention Mechanisms**: Combining traditional attention with linear attention, sparse patterns, and GQA
- **SwiGLU Feed-Forward Networks**: With Mixture-of-Experts (MoE) dynamic gating capabilities
- **RoPE Positional Encoding**: Rotary Position Embeddings for improved context handling
- **Multi-Modal Reasoning**: 8+ reasoning types with unified orchestration
- **Self-Improving World Model**: Autonomous improvement with safety constraints and kill switches
- **Production-Ready RAG**: Hierarchical memory with GraphRAG, Merkle LSM compaction, and ZK proofs
- **Mathematical Verification**: Bayesian calculation verification with learning integration
- **Local GPT Provider**: Fine-tuned model serving with streaming, perplexity, and confidence calibration

**Key Statistics:**
- GraphixExecutor: ~1,500 lines of production code
- GraphixTransformer: ~1,300 lines with tokenizer
- UnifiedReasoner: ~3,600 lines with tool selection
- HierarchicalMemory: ~1,900 lines with embeddings
- LocalGPTProvider: ~430 lines with calibration
- VULCAN Cognitive System: 285,000+ lines of code across 256 files
- Supported Reasoning Types: 8+ (Symbolic, Causal, Probabilistic, Analogical, Multimodal, Mathematical, etc.)
- Memory Systems: 10+ specialized subsystems with persistent storage

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    VULCAN INTERNAL LLM ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    APPLICATION LAYER                             │   │
│  │   Text Generation │ Reasoning │ Question Answering │ Analysis    │   │
│  └───────────────────────────┬─────────────────────────────────────┘   │
│                              │                                          │
│  ┌───────────────────────────▼─────────────────────────────────────┐   │
│  │                    UNIFIED REASONING LAYER                       │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌───────────┐  │   │
│  │  │ Symbolic   │  │ Causal     │  │ Probabilis │  │ Analogical│  │   │
│  │  │ Reasoning  │  │ Reasoning  │  │ tic        │  │ Reasoning │  │   │
│  │  └────────────┘  └────────────┘  └────────────┘  └───────────┘  │   │
│  └───────────────────────────┬─────────────────────────────────────┘   │
│                              │                                          │
│  ┌───────────────────────────▼─────────────────────────────────────┐   │
│  │                    WORLD MODEL LAYER                             │   │
│  │   Causal DAG │ Prediction Engine │ Intervention Manager          │   │
│  │   Correlation Tracker │ Self-Improvement Drive                   │   │
│  └───────────────────────────┬─────────────────────────────────────┘   │
│                              │                                          │
│  ┌───────────────────────────▼─────────────────────────────────────┐   │
│  │                    GRAPHIX TRANSFORMER CORE                      │   │
│  │  ┌────────────────────────────────────────────────────────────┐ │   │
│  │  │  Token Embeddings (RoPE) → Transformer Layers (N layers)   │ │   │
│  │  │  → Layer Norm → Output Projection → Token Prediction       │ │   │
│  │  └────────────────────────────────────────────────────────────┘ │   │
│  └───────────────────────────┬─────────────────────────────────────┘   │
│                              │                                          │
│  ┌───────────────────────────▼─────────────────────────────────────┐   │
│  │                    IR EXECUTION ENGINE                           │   │
│  │   IR Attention │ IR FeedForward │ IR LayerNorm │ IR Embeddings   │   │
│  │   KV Cache │ Sparse Patterns │ Hybrid Attention │ MoE Gating    │   │
│  └───────────────────────────┬─────────────────────────────────────┘   │
│                              │                                          │
│  ┌───────────────────────────▼─────────────────────────────────────┐   │
│  │                    MEMORY & CONTEXT LAYER                        │   │
│  │   Persistent Context │ Graph RAG │ Hierarchical Memory           │   │
│  │   Embedding Cache │ Relevance Scoring │ Compression              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Core LLM Components

### 3.1 GraphixTransformer

**Location:** `src/llm_core/graphix_transformer.py`

The GraphixTransformer is the main transformer model class that orchestrates all LLM operations. It provides:

#### Key Features:
- **IR-Based Execution**: Uses intermediate representation for flexible computation
- **LoRA Support**: Low-Rank Adaptation for efficient fine-tuning
- **Top-P Sampling**: Nucleus sampling for generation
- **Gradient Checkpointing**: Memory-efficient training
- **KV Caching**: Efficient inference through key-value caching

#### Configuration Parameters:

```python
@dataclass
class GraphixTransformerConfig:
    num_layers: int = 6              # Number of transformer layers
    hidden_size: int = 256           # Hidden dimension size
    num_heads: int = 4               # Number of attention heads
    vocab_size: int = 4096           # Vocabulary size
    max_position_embeddings: int = 1024  # Maximum sequence length
    dropout: float = 0.1             # Dropout probability
    layer_norm_eps: float = 1e-5     # Layer norm epsilon
    seed: Optional[int] = 1234       # Random seed
    gradient_checkpointing: bool = False  # Memory optimization
    dtype: str = "float32"           # Data type
    lora_rank: int = 0               # LoRA rank (0 = disabled)
    lora_alpha: float = 1.0          # LoRA scaling factor
```

#### Core Methods:

| Method | Description |
|--------|-------------|
| `forward(tokens)` | Standard forward pass through transformer |
| `generate(prompt, max_new_tokens, temperature, top_p)` | Autoregressive text generation |
| `get_embeddings(text)` | Get embeddings for input text |
| `get_logits(input)` | Get logits for next token prediction |
| `add_adapter(name, rank, alpha)` | Add LoRA adapter |
| `save(path)` / `load(path)` | Model persistence |

#### SimpleTokenizer:

The transformer includes a built-in word-level tokenizer with:
- Pre-populated vocabulary of 400+ common English words
- Special tokens: `<pad>`, `<unk>`, `<bos>`, `<eos>`
- Dynamic vocabulary expansion
- Encode/decode functionality

### 3.2 IR (Intermediate Representation) System

The IR system provides a graph-based representation of transformer operations, enabling:
- **Hardware Abstraction**: Same IR can target CPU, GPU, or custom accelerators
- **Optimization**: Graph-level optimizations like fusion and dead code elimination
- **Debugging**: Clear visibility into computation structure
- **Extensibility**: Easy to add new operations or modify existing ones

#### IR Node Types:

| Type | Description | Example |
|------|-------------|---------|
| `transform` | Linear transformations | Q/K/V projections |
| `lookup` | Embedding lookups | Token embeddings |
| `filter` | Activation functions | SwiGLU |
| `combine` | Tensor combination | Residual connections |
| `reduce_*` | Reduction operations | Mean, variance |
| `cache` | KV caching | Attention cache |
| `dropout` | Regularization | Dropout layers |
| `moe_gate` | Expert gating | MoE routing |
| `spike` | Sparse activation | Spiking gate |

### 3.3 GraphixExecutor - Production IR Engine

**Location:** `src/llm_core/graphix_executor.py`

The GraphixExecutor is the production-grade IR execution engine with comprehensive optimizations:

#### Enterprise Features:

```
✅ CORE EXECUTION
- Multi-stage IR graph execution (embeddings, attention, FFN, layer norm)
- Dynamic graph optimization and fusion
- Automatic kernel selection and optimization
- Memory-efficient execution with gradient checkpointing

✅ ADVANCED OPTIMIZATIONS
- Flash Attention implementation
- Kernel fusion (attention + feedforward)
- Mixed precision (FP16, BF16, INT8)
- Dynamic quantization and dequantization
- KV cache management with eviction policies
- Sparse attention patterns (sliding window, block-sparse)

✅ LORA & PEFT
- LoRA adapter fusion and application
- Multi-adapter support with dynamic switching
- Adapter merging and quantization
- Fine-grained adapter control per layer

✅ DISTRIBUTED & SCALING
- Tensor parallelism hints
- Pipeline parallelism support
- Expert parallelism for MoE
- Gradient accumulation

✅ OBSERVABILITY
- Performance profiling and tracing
- Memory tracking and optimization
- Execution graph visualization
- Audit logging and compliance

✅ ROBUSTNESS
- Automatic error recovery
- Fallback execution paths
- Checkpointing and state management
- Numerical stability checks
```

#### Execution Modes:

| Mode | Description |
|------|-------------|
| `TRAINING` | Full gradient computation |
| `INFERENCE` | Optimized for speed |
| `EVALUATION` | Metrics collection |
| `PROFILING` | Detailed timing |

#### Precision Modes:

| Mode | Description | Memory Savings |
|------|-------------|----------------|
| `FP32` | Full precision | Baseline |
| `FP16` | Half precision | 50% |
| `BF16` | Brain float | 50% |
| `INT8` | 8-bit quantized | 75% |
| `MIXED` | Dynamic selection | Variable |

#### KV Cache Management:

```python
class KVCacheManager:
    """Manages KV cache with eviction policies."""
    
    # Eviction Policies:
    # - LRU: Least Recently Used (default)
    # - LFU: Least Frequently Used
    # - FIFO: First In First Out
    # - ADAPTIVE: Weighted recency + frequency
    
    def get(self, layer_idx, head_idx, position) -> Optional[KVCacheEntry]
    def put(self, layer_idx, head_idx, position, keys, values) -> None
    def get_stats(self) -> Dict[str, Any]
```

#### Quantization Support:

```python
class QuantizationManager:
    """Manages quantization and dequantization."""
    
    def quantize(tensor, key) -> Tuple[List[int], float, float]:
        """Quantize tensor to INT8 with scale/zero-point."""
        
    def dequantize(quantized, scale, zero_point) -> List[float]:
        """Dequantize back to float."""
```

#### Performance Instrumentation:

```python
# Decorator for timing operations
@timed_operation("GraphixExecutor.execute")
def execute(self, graph_ir, inputs):
    ...

# Global performance stats
_PERF_STATS = {
    "operation_name": {
        "count": 0,
        "total_ms": 0.0,
        "min_ms": float("inf"),
        "max_ms": 0.0
    }
}

# Performance summary
def log_performance_summary():
    """Log hottest operations sorted by total time."""
```

#### Executor Configuration:

```python
@dataclass
class ExecutorConfig:
    mode: ExecutionMode = ExecutionMode.INFERENCE
    precision: PrecisionMode = PrecisionMode.FP32
    attention_impl: AttentionImpl = AttentionImpl.FLASH
    use_flash_attention: bool = True
    use_kernel_fusion: bool = True
    use_kv_cache: bool = True
    kv_cache_size: int = 2048
    kv_cache_eviction: CacheEvictionPolicy = CacheEvictionPolicy.LRU
    max_batch_size: int = 32
    gradient_checkpointing: bool = False
    enable_profiling: bool = False
    enable_audit: bool = True
    use_quantization: bool = False
    quantization_bits: int = 8
    compile_graphs: bool = True
    optimize_memory: bool = True
```

#### Execution Pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│                    GRAPHIX EXECUTOR PIPELINE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Inputs ──► Audit Log ──► Stage 1: Embeddings                   │
│                               │                                 │
│                               ▼                                 │
│                      Stage 2: Transformer Layers (Loop)         │
│                         ┌─────────────────────────────┐         │
│                         │ Pre-LN ──► Attention        │         │
│                         │    │         │              │         │
│                         │    │    KV Cache Lookup     │         │
│                         │    │         │              │         │
│                         │    └──► Residual            │         │
│                         │              │              │         │
│                         │ Pre-LN ──► FFN (SwiGLU)     │         │
│                         │    │         │              │         │
│                         │    └──► Residual ─────────►│         │
│                         └─────────────────────────────┘         │
│                               │                                 │
│                               ▼                                 │
│                      Stage 3: Final Layer Norm                  │
│                               │                                 │
│                               ▼                                 │
│                      Metrics Update ──► Result                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.4 Local GPT Provider

**Location:** `src/local_llm/provider/local_gpt_provider.py`

The LocalGPTProvider serves fine-tuned GPT models with production features:

#### Provider Configuration:

```python
@dataclass
class ProviderInitConfig:
    model_path: str              # Path to llm_best_model.pt
    vocab_path: str              # Path to vocab.json
    device: str = "cpu"          # cpu, cuda, cuda:0, etc.
    seq_len: int = 256           # Maximum sequence length
    dim: int = 384               # Model dimension
    n_layers: int = 6            # Number of layers
    n_heads: int = 8             # Number of attention heads
    ff_mult: int = 4             # FFN expansion factor
    dropout: float = 0.0         # Dropout rate
    dtype: str = "float32"       # float32, float16, bfloat16
    use_autocast: bool = False   # Enable automatic mixed precision
    calibration_path: Optional[str] = None  # Confidence calibration
    temperature: float = 0.9     # Sampling temperature
    top_k: int = 64              # Top-K sampling
    top_p: float = 0.95          # Nucleus sampling
    repetition_penalty: float = 1.05  # Repetition penalty
    eos_token: Optional[str] = None   # End of sequence token
```

#### API Methods:

| Method | Description |
|--------|-------------|
| `generate(prompt, ...)` | Single text generation with metadata |
| `generate_batch(prompts, ...)` | Batch generation |
| `generate_stream(prompt, ...)` | Streaming generator yielding partial text |
| `perplexity(text)` | Compute perplexity score |
| `batch_perplexity(texts)` | Batch perplexity computation |

#### Confidence Calibration:

```python
class OptionalCalibrator:
    """Remaps model confidences for better calibration."""
    
    # Supported calibration types:
    # - kind="scale": p' = scale * p + bias
    # - kind="temperature": p' = sigmoid(logit(p) / T)
    # - kind="isotonic": Piecewise linear mapping (future)
    
    def calibrate_prob(self, p: float) -> float:
        """Apply calibration to probability."""
```

#### Streaming Generation:

```python
def generate_stream(
    self,
    prompt: str,
    max_new_tokens: int = 128,
    chunk_size: int = 1,
) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
    """
    Yields (partial_text, meta) every chunk_size new tokens.
    
    Meta includes:
    - new_ids: Newly generated token IDs
    - total_len: Total sequence length
    - emitted: Total tokens emitted
    """
```

---

## 4. Attention Mechanisms

**Location:** `src/llm_core/ir_attention.py`

The attention system implements a sophisticated hybrid attention mechanism:

### Architecture Details:

```
┌─────────────────────────────────────────────────────────────────┐
│                    ATTENTION MECHANISM                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input ──► Q Projection ──┐                                     │
│                           ▼                                     │
│  Input ──► K Projection ──► Hybrid Attention ──► Softmax        │
│                           ▲       │                  │          │
│  Input ──► V Projection ──┘       │                  │          │
│                                   │                  │          │
│                                   ▼                  ▼          │
│                           KV Cache          Attention Dropout   │
│                                                      │          │
│                                                      ▼          │
│                                              Weighted Values    │
│                                                      │          │
│                                                      ▼          │
│                                              Spike Gate         │
│                                                      │          │
│                                                      ▼          │
│                                              Output Projection  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Features:

#### 1. Grouped Query Attention (GQA)
- Supports fewer KV heads than query heads
- Reduces memory bandwidth requirements
- Parameter: `num_kv_heads` (if < `num_heads`, enables GQA)

#### 2. Hybrid Attention
- Combines quadratic (softmax) and linear attention
- `linear_ratio`: 0.5 (configurable blend)
- Reduces computational complexity for long sequences

#### 3. KV Caching
- Dedicated cache nodes for K and V tensors
- Enables efficient autoregressive generation
- Cache axis: 0 (sequence dimension)

#### 4. Sparse Patterns
- Windowed attention support
- `window_size`: 128 tokens
- Pattern type: `windowed`

#### 5. Spiking Gate
- Threshold-based sparse activation
- `threshold`: 0.3
- Improves efficiency by pruning weak attention

#### 6. Causal Masking
- Built-in causal mask support
- Prevents attending to future tokens

### IR Structure:

```json
{
  "type": "attention_subgraph",
  "params": {
    "num_heads": 4,
    "hidden_size": 256,
    "num_kv_heads": null,
    "causal_masking": true,
    "sparse": "windowed",
    "window_size": 128
  },
  "metadata": {
    "semantic": "hybrid_grouped_attention",
    "inference": "kv_cache_supported",
    "sampling": {"type": "top_p", "p": 0.9}
  }
}
```

---

## 5. Feed-Forward Networks

**Location:** `src/llm_core/ir_feedforward.py`

The FFN uses a SwiGLU architecture with Mixture-of-Experts (MoE) capabilities:

### Architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                    FEED-FORWARD NETWORK (SwiGLU + MoE)          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input ──► Dynamic Gate ──┬──► Linear Gate ──┐                  │
│               (MoE)       │                  ▼                  │
│                           └──► Linear Expand ──► SwiGLU         │
│                                                    │            │
│                                                    ▼            │
│                                              Linear Project     │
│                                                    │            │
│                                                    ▼            │
│                                               Dropout           │
│                                                    │            │
│  Input ─────────────────────────────────────────► + (Residual)  │
│                                                    │            │
│                                                    ▼            │
│                                                 Output          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Features:

#### 1. SwiGLU Activation
- Combines Swish activation with gating
- Superior to standard ReLU/GELU for transformers
- Formula: `SwiGLU(x) = Swish(W_gate * x) ⊗ (W_up * x)`

#### 2. Mixture-of-Experts (MoE) Gating
- Dynamic expert selection
- Parameters:
  - `moe_experts`: 4 (number of experts)
  - `moe_top_k`: 2 (experts activated per token)
- Mode: `top_k_select`

#### 3. Scaled Residual Connection
- Residual scale: 0.5
- Improves training stability for deep networks

#### 4. Intermediate Size
- Default: `hidden_size * 4`
- Standard transformer expansion ratio

### Configuration:

```python
"params": {
    "hidden_size": 256,
    "intermediate": 1024,  # hidden_size * 4
    "dropout_p": 0.1,
    "residual_scale": 0.5,
    "moe_experts": 4,
    "moe_top_k": 2
}
```

---

## 6. Normalization Layers

**Location:** `src/llm_core/ir_layer_norm.py`

Supports multiple normalization strategies:

### Normalization Types:

#### 1. RMSNorm (Default)
- Root Mean Square normalization
- Omits mean calculation and shift parameter
- More efficient than LayerNorm
- Formula: `RMSNorm(x) = x / sqrt(mean(x²) + eps) * scale`

#### 2. Standard LayerNorm
- Full mean and variance normalization
- Includes shift (bias) parameter
- Formula: `LayerNorm(x) = (x - mean) / sqrt(var + eps) * scale + shift`

#### 3. GroupNorm (Optional)
- Groups channels for normalization
- Applied after base normalization
- Useful for certain architectural patterns

### Features:

- **Learned Epsilon Scheduling**: Adaptive epsilon for training stability
- **Pre-Norm Position**: Normalization applied before attention/FFN
- **Configurable Groups**: For GroupNorm integration

### IR Structure:

```json
{
  "type": "layer_norm_subgraph",
  "params": {
    "hidden_size": 256,
    "eps": 1e-5,
    "type": "rmsnorm",
    "groups": null,
    "eps_schedule": "learned",
    "position": "pre"
  }
}
```

---

## 7. Embedding System

**Location:** `src/llm_core/ir_embeddings.py`

### Rotary Position Embeddings (RoPE)

Unlike traditional additive position embeddings, Vulcan uses RoPE:

#### Advantages of RoPE:
- Better extrapolation to longer sequences
- Applied during Q/K projection in attention
- No separate position lookup table needed
- Preserves relative position information

#### Configuration:

```python
"params": {
    "vocab_size": 4096,
    "hidden_size": 256,
    "max_positions": 1024,
    "pos_mode": "rotary",
    "pos_base": 10000,  # Base frequency for RoPE
    "prune_vocab_below_freq": 5  # Dynamic vocab pruning hint
}
```

### Embedding Pipeline:

```
Token IDs ──► Token Lookup ──► Dropout ──► Norm Scale ──► Output
                  │
                  └── (RoPE applied later in Attention Q/K)
```

---

## 8. Persistent Context Management

**Location:** `src/llm_core/persistant_context.py`

A production-ready RAG (Retrieval-Augmented Generation) implementation:

### Architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│               PERSISTENT CONTEXT MANAGER                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Query ──► Embedding ──► Retrieval ──► Chunking ──► Scoring     │
│                              │             │           │        │
│                              ▼             ▼           ▼        │
│                         Memory System   Strategies  Relevance   │
│                              │             │           │        │
│                              └─────────────┴───────────┘        │
│                                           │                     │
│                                           ▼                     │
│                      Parent-Child Expansion ──► Reranking       │
│                                                      │          │
│                                                      ▼          │
│                                              Compression        │
│                                                      │          │
│                                                      ▼          │
│                                              Token Budget Fit   │
│                                                      │          │
│                                                      ▼          │
│                                              Final Context      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components:

#### 1. Chunking Strategies

| Strategy | Description |
|----------|-------------|
| `FIXED_SIZE` | Fixed character chunks |
| `SEMANTIC` | Sentence-boundary aware |
| `SLIDING_WINDOW` | Overlapping windows |
| `HIERARCHICAL` | Paragraphs → sentences |
| `PARAGRAPH` | Paragraph-based |

#### 2. Reranking Methods

| Method | Description |
|--------|-------------|
| `NONE` | No reranking |
| `CROSS_ENCODER` | Neural reranking |
| `RECIPROCAL_RANK_FUSION` | Multi-signal fusion |
| `DIVERSITY` | MMR-based diversity |

#### 3. Compression Methods

| Method | Description |
|--------|-------------|
| `NONE` | No compression |
| `EXTRACTIVE` | Select important sentences |
| `ABSTRACTIVE` | Use summaries |
| `HYBRID` | Mix of both |

### Configuration:

```python
@dataclass
class ContextConfig:
    max_context_tokens: int = 8192
    retrieval_k: int = 50
    rerank_top_k: int = 20
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.HIERARCHICAL
    chunk_size: int = 512
    chunk_overlap: int = 128
    reranking_method: RerankingMethod = RerankingMethod.RECIPROCAL_RANK_FUSION
    compression_method: CompressionMethod = CompressionMethod.EXTRACTIVE
    compression_ratio: float = 0.5
    use_parent_child_context: bool = True
    temporal_decay_factor: float = 0.95
    diversity_threshold: float = 0.8
    cache_size: int = 1000
```

---

## 9. Memory Architecture

### 9.1 Hierarchical Memory System

**Location:** `src/vulcan/memory/hierarchical.py`

The HierarchicalMemory class implements a multi-level memory system with tool selection history:

#### Memory Levels:

| Level | Capacity | Decay Rate | Consolidation Threshold | Use Case |
|-------|----------|------------|------------------------|----------|
| Sensory | 50 | 0.5 | 0.7 | Raw input buffer |
| Working | 20 | 0.1 | 0.6 | Active processing |
| Short-Term | 1000 | 0.05 | 0.5 | Recent memories |
| Long-Term | 100000 | 0.001 | 0.8 | Persistent storage |

#### Memory Level Data Structure:

```python
@dataclass
class MemoryLevel:
    name: str
    capacity: int
    decay_rate: float
    consolidation_threshold: float
    memories: Dict[str, Memory]
    access_queue: deque  # Track access order for LRU
    
    def add(self, memory: Memory) -> bool
    def remove_least_salient(self, n: int) -> List[Memory]
    def get_candidates_for_consolidation(self) -> List[Memory]
```

#### Embedding Models:

The system supports multiple embedding backends with automatic fallback:

```python
# Priority order:
1. Global Model Registry (singleton, shared across components)
2. SentenceTransformer models:
   - "all-MiniLM-L6-v2" (fast, 384 dimensions)
   - "all-mpnet-base-v2" (better quality, 768 dimensions)
   - "paraphrase-MiniLM-L6-v2" (alternative)
3. Hash-based fallback (128 dimensions)
```

#### Memory Operations:

| Operation | Description |
|-----------|-------------|
| `store(content, type, importance)` | Store content at appropriate level |
| `retrieve(query)` | Retrieve memories matching query |
| `forget(memory_id)` | Remove specific memory |
| `consolidate()` | Promote memories between levels |

#### Attention-Based Retrieval:

```python
class AttentionMechanism:
    """Computes attention weights for memory retrieval."""
    
    def __init__(self, hidden_dim, input_dim):
        # Learnable attention weights
        
    def compute_attention(self, query_embedding, memory_embeddings):
        """Compute attention-weighted relevance scores."""
```

#### LLM Context Integration:

```python
def retrieve_context_for_generation(self, query_tokens, max_tokens=2048):
    """
    Retrieve relevant context from memory for LLM generation.
    
    Returns merged context from:
    1. Episodic: Recent conversation (short_term level)
    2. Semantic: Relevant concepts (long_term level)
    3. Procedural: Matching patterns with tools and utility scores
    """

def store_generation(self, prompt, generated, reasoning_trace):
    """
    Store generation in memory, updating all levels:
    - Procedural: Pattern extraction from reasoning trace
    - Episodic: Full prompt/response with timestamp
    - Semantic: Extracted concepts
    """
```

### 9.2 Persistent Memory with ZK Proofs

**Location:** `src/vulcan/memory/hierarchical.py` (PersistentHierarchicalMemory)

The PersistentHierarchicalMemory extends hierarchical memory with durable storage:

#### Storage Tiers:

```
┌─────────────────────────────────────────────────────────────────┐
│                    MEMORY STORAGE TIERS                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  HOT TIER (Episodic Memory)                                     │
│  ├── In-memory deque                                            │
│  ├── Recent items (last 24 hours)                               │
│  └── Fastest access                                             │
│                                                                 │
│  WARM TIER (Local Disk)                                         │
│  ├── Pickle serialization                                       │
│  ├── Frequently accessed items                                  │
│  └── ./local_memory_cache/*.pkl                                 │
│                                                                 │
│  COLD TIER (Persistent Store)                                   │
│  ├── PackfileStore with S3/CloudFront                           │
│  ├── MerkleLSM compaction                                       │
│  └── Long-term archival                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Components:

| Component | Purpose |
|-----------|---------|
| `PackfileStore` | S3-backed persistent storage |
| `MerkleLSM` | Merkle tree + LSM compaction |
| `GraphRAG` | Semantic retrieval augmented generation |
| `UnlearningEngine` | Privacy-preserving data deletion |
| `ZKProver` | Zero-knowledge proofs for verification |
| `LearningStatePersistence` | Learning state recovery |

#### Interaction Storage:

```python
def store_interaction(
    self,
    query_id: str,
    query: str,
    answer: str,
    tools_used: Optional[List[str]] = None,
    success: bool = True,
    latency_ms: float = 0.0,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Store interaction across all tiers:
    1. GraphRAG for semantic retrieval
    2. Episodic memory for quick access
    3. LearningStatePersistence for recovery
    """
```

### 9.3 Tool Selection Memory

The HierarchicalMemory includes specialized storage for tool selection decisions:

#### Tool Selection Record:

```python
@dataclass
class ToolSelectionRecord:
    record_id: str
    timestamp: float
    problem_features: np.ndarray
    problem_description: str
    selected_tools: List[str]
    execution_strategy: str
    performance_metrics: Dict[str, float]  # latency, accuracy, energy
    context: Dict[str, Any]
    success: bool
    utility_score: float
    metadata: Dict[str, Any]
```

#### Problem Pattern Discovery:

```python
@dataclass
class ProblemPattern:
    pattern_id: str
    feature_signature: np.ndarray  # Centroid of similar problems
    typical_tools: List[str]       # Most successful tools
    success_rate: float            # Historical success rate
    avg_utility: float             # Average utility score
    occurrence_count: int          # Number of matching problems
    examples: List[str]            # Sample problem IDs
```

#### Tool Recommendation:

```python
def get_recommended_tools(
    self,
    problem_features: np.ndarray,
    problem_description: Optional[str] = None,
    max_recommendations: int = 5,
) -> List[Dict[str, Any]]:
    """
    Get tool recommendations based on:
    1. Similar successful problems
    2. Discovered problem patterns
    
    Returns recommendations with:
    - tool: Tool name
    - confidence: Utility-based confidence
    - success_rate: Historical success rate
    - occurrence_count: How often used
    - pattern_match: Whether matches a known pattern
    - score: Combined ranking score
    """
```

#### Background Tasks:

The system runs two background threads:

1. **Consolidation Thread**: Periodically promotes memories between levels
2. **Pattern Mining Thread**: Discovers problem patterns from tool selection history

```python
# Pattern mining algorithm:
1. Group records by feature similarity (threshold: 0.8)
2. For each group with >= 3 records:
   - Compute centroid feature signature
   - Calculate success rate and average utility
   - Identify typical tools from successful records
   - Create ProblemPattern
```

---

## 10. Reasoning Systems

### 10.1 Unified Reasoning Interface

**Location:** `src/vulcan/reasoning/unified_reasoning.py`

The UnifiedReasoner orchestrates all reasoning capabilities with production-grade features:

#### Initialization Parameters:

```python
class UnifiedReasoner:
    def __init__(
        self,
        enable_learning: bool = True,     # Enable continuous learning
        enable_safety: bool = True,       # Enable safety validation
        max_workers: Optional[int] = None,  # Thread pool size (env: VULCAN_MAX_WORKERS)
        config: Optional[Dict[str, Any]] = None,
    ):
```

#### Reasoning Strategies:

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `SEQUENTIAL` | Execute reasoning types in sequence | Simple problems |
| `PARALLEL` | Execute reasoning types concurrently | Independent analyses |
| `ENSEMBLE` | Weighted voting across reasoners | High confidence needed |
| `HIERARCHICAL` | Dependency-based execution | Complex reasoning chains |
| `ADAPTIVE` | Dynamic strategy selection | Unknown problem types |
| `HYBRID` | Custom combination approach | Specialized pipelines |
| `PORTFOLIO` | Complementary reasoner selection | Diverse perspectives |
| `UTILITY_BASED` | Utility-optimized selection | Resource-constrained |

#### Singleton Tool Weight Manager:

```python
class ToolWeightManager:
    """Singleton manager for tool weights shared between Learning and Ensemble.
    
    BUG #5 FIX: Learning system was updating tool weights in its own dictionary,
    but Ensemble was reading from a separate dictionary. Weights never propagated.
    This singleton ensures both systems use the same weight storage.
    """
    
    def get_weight(self, tool: str, default: float = 1.0) -> float
    def set_weight(self, tool: str, value: float) -> None
    def adjust_weight(self, tool: str, delta: float) -> None
    def get_all_weights(self, tools: List[str]) -> Dict[str, float]

# Usage:
get_weight_manager().adjust_weight("causal", 0.01)
```

#### Environment Detection:

```python
def _is_test_environment() -> bool:
    """
    Determine if running in test environment.
    
    Bug #3 Fix: Default to PRODUCTION for safety.
    Only return True if explicitly in test mode.
    
    Explicit TEST indicators:
    - PYTEST_CURRENT_TEST is set
    - VULCAN_TEST_MODE=true
    - VULCAN_ENV=test
    
    Explicit PRODUCTION indicators (override test):
    - VULCAN_PRODUCTION=true
    - VULCAN_FORCE_PRODUCTION_REASONING=true
    - VULCAN_ENV=production
    - ENVIRONMENT=production
    """
```

#### Utility Context:

```python
@dataclass
class UtilityContext:
    mode: ContextMode      # RUSH, ACCURATE, EFFICIENT, BALANCED
    time_budget: float     # Milliseconds
    energy_budget: float   # Millijoules
    min_quality: float     # Minimum acceptable quality
    max_risk: float        # Maximum acceptable risk
    user_preferences: Dict[str, Any]
```

#### Main Reasoning Method:

```python
def reason(
    self,
    input_data: Any,
    query: Optional[Dict[str, Any]] = None,
    reasoning_type: Optional[ReasoningType] = None,
    strategy: ReasoningStrategy = ReasoningStrategy.ADAPTIVE,
    confidence_threshold: Optional[float] = None,
    constraints: Optional[Dict[str, Any]] = None,
) -> ReasoningResult:
    """
    Enhanced reasoning interface with production tool selection.
    
    Pipeline:
    1. Shutdown check
    2. Create reasoning chain
    3. Safety input validation
    4. Distribution shift detection
    5. Reasoning type determination
    6. VOI (Value of Information) gate
    7. Plan optimization
    8. Tool selection for portfolio/utility strategies
    9. Strategy execution with timeout
    10. Confidence calibration
    11. Post-processing with mathematical verification
    12. Safety output validation
    13. Learning update
    14. Metrics and caching
    """
```

### 10.2 Reasoning Types

#### Available Reasoners:

| Type | Class | Description |
|------|-------|-------------|
| `PROBABILISTIC` | `ProbabilisticReasoner` | Uncertainty quantification |
| `SYMBOLIC` | `SymbolicReasoner` | Logic and formal reasoning |
| `CAUSAL` | `EnhancedCausalReasoning` | Cause-effect relationships |
| `ANALOGICAL` | `AnalogicalReasoner` | Pattern matching, similarity |
| `MULTIMODAL` | `MultiModalReasoningEngine` | Cross-modal reasoning |
| `MATHEMATICAL` | `MathematicalComputationTool` | Computation and proofs |
| `COUNTERFACTUAL` | `CounterfactualReasoner` | What-if scenarios |
| `LANGUAGE` | `LanguageReasoner` | Natural language understanding |

#### Specialized Reasoners:

```python
# Counterfactual reasoning (from causal reasoner)
self.counterfactual = CounterfactualReasoner(causal_reasoner)

# Cross-modal reasoning
self.cross_modal = CrossModalReasoner()

# Multimodal with registered modality reasoners
self.multimodal = MultiModalReasoningEngine(enable_learning=enable_learning)
self.multimodal.register_modality_reasoner(ModalityType.TEXT, symbolic_reasoner)
self.multimodal.register_modality_reasoner(ModalityType.CODE, symbolic_reasoner)
```

#### Ensemble Weighting:

```python
def _ensemble_reasoning(self, plan, reasoning_chain) -> ReasoningResult:
    """Ensemble reasoning with weighted voting."""
    
    for reasoning_type, result in results:
        # Base weight from confidence
        base_weight = result.confidence
        
        # Type weight from historical performance
        type_weight = self._get_reasoning_type_weight(reasoning_type)
        
        # Utility weight if context available
        if utility_context:
            execution_time_ms = result.metadata.get("execution_time_ms", 100)
            utility_weight = self._calculate_result_utility(
                result, utility_context, execution_time_ms
            )
            weights.append(base_weight * type_weight * utility_weight)
        else:
            weights.append(base_weight * type_weight)
    
    # Issue #53 Fix: Handle zero weights gracefully
    if sum(weights) <= 0:
        weights = [1.0 / len(results)] * len(results)
```

### 10.3 Mathematical Verification

**Location:** `src/vulcan/reasoning/mathematical_verification.py`

The MathematicalVerificationEngine validates calculation results:

#### Verification Types:

```python
class MathVerificationStatus(Enum):
    VERIFIED = "verified"
    ERROR_DETECTED = "error_detected"
    UNCERTAIN = "uncertain"
    NOT_APPLICABLE = "not_applicable"
```

#### Bayesian Calculation Verification:

```python
@dataclass
class BayesianProblem:
    prior: float           # P(H)
    sensitivity: float     # P(E|H)
    specificity: float     # P(¬E|¬H)

def verify_bayesian_calculation(
    self,
    problem: BayesianProblem,
    computed_posterior: float,
) -> VerificationResult:
    """
    Verify Bayesian calculation using Bayes' theorem:
    P(H|E) = P(E|H) * P(H) / P(E)
    
    where P(E) = P(E|H)*P(H) + P(E|¬H)*P(¬H)
    """
```

#### Arithmetic Verification:

```python
def verify_arithmetic(
    self,
    expression: str,
    computed_result: float,
    variables: Dict[str, float],
) -> VerificationResult:
    """Verify arithmetic expression evaluation."""
```

#### Learning Integration:

```python
# Constants for learning feedback
MATH_VERIFICATION_CONFIDENCE_BOOST = 1.1  # Boost for verified correct
MATH_ERROR_CONFIDENCE_PENALTY = 0.5       # Penalty for error detected
MATH_ACCURACY_REWARD = 0.015              # Learning reward for accuracy
MATH_ACCURACY_PENALTY = -0.01             # Learning penalty for errors
MATH_WEIGHT_ADJUSTMENT_PENALTY = -0.01    # Tool weight adjustment

def _apply_verification_to_result(self, result, verification, task):
    """Apply verification results and update learning system."""
    
    if verification.status == MathVerificationStatus.VERIFIED:
        # Boost confidence
        result.confidence *= MATH_VERIFICATION_CONFIDENCE_BOOST
        
        # Reward tool through learning
        if self._math_accuracy_integration:
            self._math_accuracy_integration.reward_tool(tool_name, self.learner)
            
    elif verification.status == MathVerificationStatus.ERROR_DETECTED:
        # Apply corrections
        result.conclusion['math_correction'] = {
            'original': result.conclusion,
            'corrected': verification.corrections.get('correct_result'),
            'errors': [e.value for e in verification.errors],
        }
        
        # Reduce confidence
        result.confidence *= MATH_ERROR_CONFIDENCE_PENALTY
        
        # Penalize tool through learning
        for error in verification.errors:
            self._math_accuracy_integration.penalize_tool(tool_name, error, self.learner)
        
        # Update shared weight manager
        get_weight_manager().adjust_weight(tool_name, MATH_WEIGHT_ADJUSTMENT_PENALTY)
```

---

## 11. World Model Integration

**Location:** `src/vulcan/world_model/`

The World Model provides the cognitive foundation for understanding and predicting the environment.

### 11.1 Core Components

| Component | File | Description |
|-----------|------|-------------|
| `world_model_core.py` | Core world model | Main orchestration |
| `causal_graph.py` | Causal DAG | Directed acyclic graph of causes |
| `correlation_tracker.py` | Correlation tracking | Statistical relationships |
| `prediction_engine.py` | Prediction | Ensemble prediction with uncertainty |
| `intervention_manager.py` | Interventions | Prioritized intervention scheduling |
| `dynamics_model.py` | Dynamics | Temporal state evolution |
| `confidence_calibrator.py` | Confidence | Probability calibration |
| `invariant_detector.py` | Invariants | Stable pattern detection |
| `world_model_router.py` | Routing | Update routing and prioritization |

#### Causal DAG:

```python
class CausalGraph:
    """Directed acyclic graph of causal relationships."""
    
    # Evidence types for edge creation
    evidence_types = ["observational", "intervention", "linguistic"]
    
    # Methods
    def add_edge(self, cause, effect, strength, evidence_type)
    def remove_edge(self, cause, effect)
    def get_causal_path(self, start, end) -> List[str]
    def detect_cycles(self) -> List[List[str]]
    def break_cycles(self) -> int
```

#### Prediction Engine:

```python
class PredictionEngine:
    """Ensemble prediction with uncertainty quantification."""
    
    def predict(self, variables, context) -> Prediction:
        """
        Returns:
        - value: Predicted value
        - confidence: Calibrated confidence score
        - uncertainty: Uncertainty estimate
        - causal_paths: Relevant causal paths used
        """
```

#### Intervention Manager:

```python
class InterventionManager:
    """Prioritized intervention scheduling with safety validation."""
    
    def schedule_intervention(self, intervention) -> bool:
        """
        Validates safety constraints before scheduling.
        Returns True if intervention was scheduled.
        """
    
    def execute_pending(self) -> List[InterventionResult]:
        """Execute scheduled interventions in priority order."""
```

### 11.2 Meta-Reasoning

**Location:** `src/vulcan/world_model/meta_reasoning/`

The meta-reasoning system enables self-reflection and improvement:

#### EXAMINE → SELECT → APPLY → REMEMBER Pattern:

```
┌─────────────────────────────────────────────────────────────────┐
│                    COGNITIVE CYCLE                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. EXAMINE                                                     │
│     - Validate and analyze observations                         │
│     - Extract variables and patterns                            │
│     - Detect interventions and anomalies                        │
│     - Analyze characteristics (complexity, uncertainty)         │
│                                                                 │
│  2. SELECT                                                      │
│     - Route updates through WorldModelRouter                    │
│     - Prioritize based on constraints                           │
│     - Choose optimal execution plan                             │
│     - Apply VOI (Value of Information) gating                   │
│                                                                 │
│  3. APPLY                                                       │
│     - Execute selected updates                                  │
│     - Update causal graph                                       │
│     - Run interventions if scheduled                            │
│     - Track correlations                                        │
│                                                                 │
│  4. REMEMBER                                                    │
│     - Update state counters                                     │
│     - Validate consistency                                      │
│     - Persist changes                                           │
│     - Update learning weights                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 12. Self-Improvement System

**Location:** `src/vulcan/world_model/meta_reasoning/`

### Self-Improvement Drive:

The autonomous self-improvement system includes:

#### Trigger Types:
- Startup initialization
- Error detection
- Performance degradation
- Idle period optimization
- Resource availability

#### Execution Pipeline:

```
1. Context Building ──► 2. Trigger Check ──► 3. Action Generation
        │                      │                      │
        ▼                      ▼                      ▼
   Performance          Self-Improvement         LLM-Driven
   Metrics              Drive Evaluation         Code Generation
        │                      │                      │
        └──────────────────────┴──────────────────────┘
                               │
                               ▼
4. AST Validation ──► 5. Diff Application ──► 6. Git Commit
        │                      │                    │
        ▼                      ▼                    ▼
   Syntax Check           File System          Version Control
        │                   Update                  │
        └──────────────────────┴──────────────────┘
                               │
                               ▼
                    7. Learning/Outcome Recording
```

#### Safety Controls:
- Kill switch: `VULCAN_ENABLE_SELF_IMPROVEMENT=0`
- Auto-commit disable: `VULCAN_SELF_IMPROVEMENT_AUTO_COMMIT=false`
- Configurable check interval (default: 24 hours)

---

## 13. Safety and Governance

### Safety Layers:

1. **Input Validation**
   - Sanitization of user inputs
   - Pattern-based threat detection
   - Content filtering

2. **Reasoning Validation**
   - SafetyAwareReasoning wrapper
   - Confidence threshold filtering
   - Output safety checks

3. **Causal Safety**
   - Causal edge validation
   - Intervention safety checks
   - Path validation

4. **Self-Improvement Safety**
   - AST validation before code application
   - Git-based rollback capability
   - Manual approval options

### Governance Features:

- **Trust-Weighted Consensus**: For proposals and changes
- **Audit Trail**: Comprehensive logging of all operations
- **CSIU Framework**: Communication, Safety, Identity, Utility constraints
- **Alignment Validation**: Motivational introspection for proposal alignment

---

## 14. Performance Instrumentation

**Location:** `src/llm_core/graphix_executor.py`

### Timing Decorator:

```python
def timed_operation(operation_name: str) -> Callable:
    """Decorator to time operations and log performance."""
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            
            # Update global stats
            stats = _PERF_STATS[operation_name]
            stats["count"] += 1
            stats["total_ms"] += elapsed_ms
            stats["min_ms"] = min(stats["min_ms"], elapsed_ms)
            stats["max_ms"] = max(stats["max_ms"], elapsed_ms)
            
            # Log slow operations (>50ms)
            if elapsed_ms > 50.0:
                logger.debug("[PERF] %s: %.1fms", operation_name, elapsed_ms)
            
            return result
        return wrapper
    return decorator
```

### Performance Statistics API:

```python
# Get all performance stats
stats = get_performance_stats()
# Returns: {"op_name": {"count": N, "total_ms": T, "avg_ms": A, "min_ms": M, "max_ms": X}}

# Reset stats
reset_performance_stats()

# Log summary (sorted by total time, hottest first)
log_performance_summary()
```

### Execution Metrics:

```python
@dataclass
class ExecutionMetrics:
    total_executions: int = 0
    total_tokens_processed: int = 0
    total_time_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    memory_peak_mb: float = 0.0
    flops_total: float = 0.0
    layer_times: Dict[str, float]
    
    def get_avg_time_per_token(self) -> float
    def get_cache_hit_rate(self) -> float
    def to_dict(self) -> Dict[str, Any]
```

### Layer-Level Profiling:

```python
# Fine-grained layer timing (logged when total > 200ms)
timings = {
    "ln1_ms": ...,           # Pre-attention layer norm
    "attention_ms": ...,     # Attention computation
    "residual1_ms": ...,     # First residual connection
    "ln2_ms": ...,           # Pre-FFN layer norm
    "ffn_ms": ...,           # Feed-forward network
    "residual2_ms": ...,     # Second residual connection
}

# FFN sub-operation timing (logged when total > 100ms)
ffn_timings = {
    "gate_proj_ms": ...,     # Gate projection
    "up_proj_ms": ...,       # Up projection
    "swiglu_ms": ...,        # SwiGLU activation
    "down_proj_ms": ...,     # Down projection
}
```

### Profiler Integration:

```python
class PerformanceProfiler:
    """Profiles execution performance."""
    
    def __init__(self, enabled: bool = False):
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.memory_snapshots: List[Tuple[str, float]] = []
        
    def start_timer(self, name: str) -> float
    def end_timer(self, name: str, start_time: float) -> float
    def record_memory(self, name: str, bytes_used: float) -> None
    def get_summary(self) -> Dict[str, Any]
```

### Audit Logging:

```python
class AuditLogger:
    """Logs execution for compliance and debugging."""
    
    def log(self, event_type: str, details: Dict[str, Any]):
        entry = {
            "timestamp": time.time(),
            "type": event_type,
            "details": details
        }
        # Append to in-memory list and optionally write to file
```

---

## 15. Configuration and Tuning

### Environment Variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `VULCAN_MAX_WORKERS` | Thread pool size | 2 |
| `VULCAN_TEST_MODE` | Enable test mode | false |
| `VULCAN_PRODUCTION` | Force production mode | false |
| `VULCAN_ENV` | Environment (test/production) | - |
| `VULCAN_ENABLE_SELF_IMPROVEMENT` | Enable self-improvement | 1 |
| `SELF_IMPROVEMENT_INTERVAL` | Check interval (seconds) | 86400 |
| `VULCAN_LLM_API_KEY` | External LLM API key | - |
| `VULCAN_FORCE_PRODUCTION_REASONING` | Force production reasoning | false |

### Model Size Guidelines:

| Configuration | Layers | Hidden | Heads | Params (Est.) | Memory |
|---------------|--------|--------|-------|---------------|--------|
| Small | 3 | 128 | 4 | ~1M | ~4MB |
| Medium | 6 | 512 | 8 | ~30M | ~120MB |
| Large | 12 | 768 | 12 | ~120M | ~480MB |
| Default | 6 | 256 | 4 | ~10M | ~40MB |

### Tuning Guidelines:

#### For Faster Inference:
```python
config = ExecutorConfig(
    mode=ExecutionMode.INFERENCE,
    precision=PrecisionMode.FP16,
    attention_impl=AttentionImpl.FLASH,
    use_kv_cache=True,
    use_kernel_fusion=True,
)
```

#### For Better Quality:
```python
config = GraphixTransformerConfig(
    num_layers=12,
    hidden_size=768,
    num_heads=12,
)

# Generation settings
model.generate(
    prompt,
    temperature=0.7,
    top_p=0.9,
    max_new_tokens=256
)
```

#### For Memory Efficiency:
```python
config = GraphixTransformerConfig(
    gradient_checkpointing=True,
    lora_rank=8,  # Use LoRA instead of full fine-tuning
)

executor_config = ExecutorConfig(
    use_quantization=True,
    quantization_bits=8,
    optimize_memory=True,
)
```

---

## 16. Code Examples

### Basic Text Generation:

```python
from src.llm_core.graphix_transformer import GraphixTransformer, GraphixTransformerConfig

# Create model
config = GraphixTransformerConfig(
    num_layers=6,
    hidden_size=256,
    num_heads=4,
    vocab_size=4096,
)
model = GraphixTransformer(config)

# Generate text
text = model.generate(
    prompt="The future of AI is",
    max_new_tokens=50,
    temperature=0.8,
    top_p=0.9,
)
print(text)
```

### Using LocalGPTProvider:

```python
from src.local_llm.provider.local_gpt_provider import LocalGPTProvider, ProviderInitConfig

# Initialize from config
provider = LocalGPTProvider.from_config_file("provider_config.json")

# Single generation
text, meta = provider.generate(
    prompt="Explain quantum computing",
    max_new_tokens=200,
    temperature=0.9,
)

# Streaming generation
for partial_text, meta in provider.generate_stream(
    prompt="Write a story about",
    max_new_tokens=500,
    chunk_size=10,
):
    print(partial_text, end="", flush=True)

# Calculate perplexity
ppl = provider.perplexity("The quick brown fox jumps over the lazy dog.")
print(f"Perplexity: {ppl:.2f}")
```

### Unified Reasoning:

```python
from src.vulcan.reasoning.unified_reasoning import UnifiedReasoner, ReasoningStrategy
from src.vulcan.reasoning.reasoning_types import ReasoningType

# Initialize reasoner
reasoner = UnifiedReasoner(
    enable_learning=True,
    enable_safety=True,
)

# Reason with specific type
result = reasoner.reason(
    input_data={"observation": "A causes B"},
    query={"question": "What happens if A occurs?"},
    reasoning_type=ReasoningType.CAUSAL,
)

# Reason with ensemble strategy
result = reasoner.reason(
    input_data="Is the hypothesis valid?",
    strategy=ReasoningStrategy.ENSEMBLE,
    confidence_threshold=0.7,
)

print(f"Conclusion: {result.conclusion}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Explanation: {result.explanation}")
```

### Hierarchical Memory:

```python
from src.vulcan.memory.hierarchical import HierarchicalMemory
from src.vulcan.memory.base import MemoryConfig, MemoryType, MemoryQuery

# Initialize memory
config = MemoryConfig(max_long_term=10000)
memory = HierarchicalMemory(config)

# Store memory
mem = memory.store(
    content="Important fact about machine learning",
    memory_type=MemoryType.SEMANTIC,
    importance=0.8,
)

# Retrieve similar memories
query = MemoryQuery(
    query_type="semantic_search",
    content="machine learning facts",
    limit=10,
)
result = memory.retrieve(query)

for mem, score in zip(result.memories, result.scores):
    print(f"[{score:.2f}] {mem.content}")

# Store tool selection
record = memory.store_tool_selection(
    problem_features=np.random.randn(128),
    problem_description="Classification task",
    selected_tools=["probabilistic", "symbolic"],
    execution_strategy="ensemble",
    performance_metrics={"latency_ms": 150, "accuracy": 0.95},
    success=True,
    utility_score=0.85,
)

# Get tool recommendations
recommendations = memory.get_recommended_tools(
    problem_features=np.random.randn(128),
    max_recommendations=5,
)
```

### GraphixExecutor Direct Usage:

```python
from src.llm_core.graphix_executor import (
    GraphixExecutor, ExecutorConfig, ExecutionMode, PrecisionMode
)

# Configure executor
config = ExecutorConfig(
    mode=ExecutionMode.INFERENCE,
    precision=PrecisionMode.FP16,
    use_flash_attention=True,
    use_kv_cache=True,
    enable_profiling=True,
)

# Create executor
executor = GraphixExecutor(
    hidden_size=256,
    num_layers=6,
    num_heads=4,
    vocab_size=4096,
    config=config,
)

# Execute IR graph
result = executor.execute(
    graph_ir={"embedding": {}, "layers": []},
    inputs={"tokens": [1, 2, 3, 4, 5]},
)

print(f"Execution time: {result['execution_time_ms']:.2f}ms")
print(f"Cache stats: {result['cache_stats']}")

# Get performance stats
stats = executor.get_stats()
print(f"Total executions: {stats['metrics']['total_executions']}")
```

---

## 17. Future Directions

## 17. Future Directions

### Planned Enhancements:

1. **Advanced Attention Patterns**
   - Multi-scale attention with hierarchical windows
   - Adaptive sparse patterns based on content
   - Memory-efficient long context (>100K tokens)
   - Sliding window with global attention tokens

2. **Enhanced MoE (Mixture of Experts)**
   - Dynamic expert allocation based on input
   - Hierarchical expert organization
   - Expert specialization and routing optimization
   - Load balancing across experts

3. **Improved Generation**
   - Speculative decoding for faster inference
   - Beam search with diversity constraints
   - Constrained generation with grammar rules
   - Multi-modal generation (text + code + structured)

4. **Distributed Execution**
   - Model parallelism across GPUs
   - Pipeline parallelism for large models
   - Distributed inference with load balancing
   - Federated learning support

5. **Safety Improvements**
   - Formal verification integration
   - Constitutional AI methods
   - Interpretability and attribution tools
   - Red team testing automation

6. **Learning Enhancements**
   - Online learning from user feedback
   - Curriculum learning for complex tasks
   - Meta-learning for few-shot adaptation
   - Reinforcement learning from human feedback (RLHF)

7. **Memory System Evolution**
   - Hierarchical attention over memory
   - Episodic memory retrieval with temporal reasoning
   - Semantic memory with knowledge graph integration
   - Procedural memory for skill acquisition

8. **Quantization and Efficiency**
   - INT4 and INT2 quantization
   - Structured pruning
   - Knowledge distillation
   - Neural architecture search (NAS)

---

## Related Documentation

- [ARCHITECTURE_OVERVIEW.md](ARCHITECTURE_OVERVIEW.md) - Platform architecture
- [EXECUTION_ENGINE.md](EXECUTION_ENGINE.md) - Execution mechanics
- [COMPLETE_SERVICE_CATALOG.md](COMPLETE_SERVICE_CATALOG.md) - Service documentation
- [GOVERNANCE.md](GOVERNANCE.md) - Governance documentation
- [SECURITY.md](SECURITY.md) - Security architecture
- [AI_TRAINING_GUIDE.md](AI_TRAINING_GUIDE.md) - Training guide
- [AUTONOMOUS_LEARNING_ANALYSIS.md](AUTONOMOUS_LEARNING_ANALYSIS.md) - Learning analysis

---

## Glossary

| Term | Definition |
|------|------------|
| **IR** | Intermediate Representation - graph-based representation of computations |
| **RoPE** | Rotary Position Embeddings - relative positional encoding method |
| **SwiGLU** | Swish-Gated Linear Unit - activation function combining Swish and gating |
| **MoE** | Mixture of Experts - sparse model with multiple expert networks |
| **GQA** | Grouped Query Attention - attention with fewer KV heads than query heads |
| **KV Cache** | Key-Value Cache - stores previous attention keys/values for efficient generation |
| **LoRA** | Low-Rank Adaptation - parameter-efficient fine-tuning method |
| **Flash Attention** | Memory-efficient attention implementation with tiling |
| **GraphRAG** | Graph Retrieval Augmented Generation - semantic retrieval with graph structure |
| **ZK Proofs** | Zero-Knowledge Proofs - cryptographic verification without revealing data |
| **VOI** | Value of Information - decision-theoretic measure for information gathering |

---

**Document Version:** 2.0.0  
**Last Updated:** December 30, 2024  
**Author:** Vulcan Deep Dive Analysis
